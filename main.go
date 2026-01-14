package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	openai "github.com/sashabaranov/go-openai"
	"gopkg.in/yaml.v3"
)

type OllamaMessage struct {
	Role      string           `json:"role"`
	Content   string           `json:"content"`
	Thinking  string           `json:"thinking,omitempty"`
	ToolCalls []OllamaToolCall `json:"tool_calls,omitempty"`
}

func (c *OllamaMessage) ToOpenAi() openai.ChatCompletionMessage {
	var toolCalls []openai.ToolCall
	for _, toolCall := range c.ToolCalls {
		toolCalls = append(toolCalls, toolCall.ToOpenAi())
	}

	return openai.ChatCompletionMessage{
		Role:             c.Role,
		Content:          c.Content,
		ToolCalls:        toolCalls,
		ToolCallID:       "", // TODO: should be set when role=tool, must match the "id" returned as part of a Tool Call from the LLM
		ReasoningContent: c.Thinking,
	}
}

type OllamaToolCall struct {
	Function OllamaFunctionCall `json:"function"`
}

func (c *OllamaToolCall) ToOpenAi() openai.ToolCall {
	return openai.ToolCall{
		Type:     "function",
		Function: c.Function.ToOpenAi(),
	}
}

type OllamaFunctionCall struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

func (c *OllamaFunctionCall) ToOpenAi() openai.FunctionCall {
	return openai.FunctionCall{
		Name:      c.Name,
		Arguments: string(c.Arguments),
	}
}

var modelFilter map[string]struct{}

func loadModelFilter(path string) (map[string]struct{}, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	filter := make(map[string]struct{})

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line != "" {
			filter[line] = struct{}{}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return filter, nil
}

func parseChoices(choices []openai.ChatCompletionChoice) (string, string, []map[string]interface{}) {
	content := ""
	thinking := ""
	var parsedToolCalls []map[string]interface{}

	if len(choices) > 0 {
		msg := choices[0].Message

		toolCalls := msg.ToolCalls
		if len(toolCalls) > 0 {
			for _, tc := range toolCalls {
				// Parse arguments using YAML to be more foregiving with improper JSON
				var argsMap map[string]interface{}
				if err := yaml.Unmarshal([]byte(tc.Function.Arguments), &argsMap); err == nil {
					parsedToolCall := map[string]interface{}{
						"function": map[string]interface{}{
							"name":      tc.Function.Name,
							"arguments": argsMap,
						},
					}
					parsedToolCalls = append(parsedToolCalls, parsedToolCall)
				} else {
					slog.Error("Failed to parse arguments for tool call", "Error", err)
				}
			}
		}

		content = msg.Content
		thinking = msg.ReasoningContent
	}

	return content, thinking, parsedToolCalls
}

func main() {
	r := gin.Default()
	// Load the API key from environment variables or command-line arguments.
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		if len(os.Args) > 1 {
			apiKey = os.Args[len(os.Args)-1]
		} else {
			slog.Error("OPENAI_API_KEY environment variable or command-line argument not set.")
			return
		}
	}

	// Try to load the API key from environment variables or command-line arguments
	baseUrl := os.Getenv("OPENAI_BASE_URL")
	if baseUrl == "" {
		if len(os.Args) > 2 {
			baseUrl = os.Args[1]
		} else {
			baseUrl = "https://openrouter.ai/api/v1/"
		}
	}

	provider := NewOpenrouterProvider(baseUrl, apiKey)

	filter, err := loadModelFilter("models-filter")
	if err != nil {
		if os.IsNotExist(err) {
			slog.Info("models-filter file not found. Skipping model filtering.")
			modelFilter = make(map[string]struct{})
		} else {
			slog.Error("Error loading models filter", "Error", err)
			return
		}
	} else {
		modelFilter = filter
		slog.Info("Loaded models from filter:")
		for model := range modelFilter {
			slog.Info(" - " + model)
		}
	}

	r.GET("/", func(c *gin.Context) {
		c.String(http.StatusOK, "Ollama is running")
	})
	r.HEAD("/", func(c *gin.Context) {
		c.String(http.StatusOK, "")
	})

	r.GET("/api/tags", func(c *gin.Context) {
		models, err := provider.GetModels(c)
		if err != nil {
			slog.Error("Error getting models", "Error", err)
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		filter := modelFilter
		// Construct a new array of model objects with extra fields
		newModels := make([]map[string]interface{}, 0, len(models))
		for _, m := range models {
			// If the filter is empty, skip the check and include all models
			if len(filter) > 0 {
				if _, ok := filter[m.Model]; !ok {
					continue
				}
			}
			newModels = append(newModels, map[string]interface{}{
				"name":        m.Name,
				"model":       m.Model,
				"modified_at": m.ModifiedAt,
				"size":        int64(270898672), // Ensure this is explicitly cast to int64
				"digest":      "9077fe9d2ae1a4a41a868836b56b8163731a8fe16621397028c2c76f838c6907",
				"details":     m.Details, // If m.Details is nil, initialize it as an empty map
			})
		}

		c.JSON(http.StatusOK, gin.H{"models": newModels})
	})

	r.POST("/api/show", func(c *gin.Context) {
		var request map[string]string
		if err := c.BindJSON(&request); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid JSON payload"})
			return
		}

		modelName := request["name"]
		if modelName == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Model name is required"})
			return
		}

		details, err := provider.GetModelDetails(modelName)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		c.JSON(http.StatusOK, details)
	})

	r.POST("/api/chat", func(c *gin.Context) {
		var request struct {
			Model     string                 `json:"model"`
			Messages  []OllamaMessage        `json:"messages"`
			Tools     []openai.Tool          `json:"tools"`
			Stream    *bool                  `json:"stream"`
			Think     *bool                  `json:"think"`
			KeepAlive string                 `json:"keep_alive"` // ex: 30.0s
			Options   map[string]interface{} `json:"options"`    // ex: {"num_ctx": 4096.0}
		}

		// Parse the JSON request
		bodyBytes, _ := c.GetRawData()

		//slog.Info("Request", "Request", string(bodyBytes))

		if err := json.Unmarshal(bodyBytes, &request); err != nil {
			//if err := c.ShouldBindJSON(&request); err != nil {
			// Read the raw request body as a string for logging
			slog.Error("Invalid JSON payload", "Error", err, "RequestBody", string(bodyBytes))

			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid JSON payload"})
			return
		}

		var openAiMessages []openai.ChatCompletionMessage
		for _, message := range request.Messages {
			openAiMessages = append(openAiMessages, message.ToOpenAi())
		}

		//slog.Info("Requested model", "model", request.Model)
		fullModelName, err := provider.GetFullModelName(c, request.Model)
		if err != nil {
			slog.Error("Error getting full model name", "Error", err, "model", request.Model)
			// Ollama returns 404 for an incorrect model name
			c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
			return
		}

		// Determine if streaming is needed (defaults to true if not specified for /api/chat)
		// IMPORTANT: Open WebUI may NOT send "stream": true for /api/chat, implying it.
		// Need to check what request the Open WebUI sends. If it doesn't send it, default to true.
		streamRequested := true
		if request.Stream != nil {
			streamRequested = *request.Stream
		}

		// If streaming is not requested, separate logic is required to gather the full response and send it as one JSON.
		// For now, only streaming is implemented.
		if !streamRequested {
			// Handle non-streaming response

			req := openai.ChatCompletionRequest{
				Model:    request.Model,
				Messages: openAiMessages,
				Tools:    request.Tools,
				Stream:   false,
			}

			// Call Chat to get the complete response
			response, err := provider.Chat(c, req)
			if err != nil {
				slog.Error("Failed to get chat response", "Error", err)
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}

			// Format the response according to Ollama's format
			if len(response.Choices) == 0 {
				c.JSON(http.StatusInternalServerError, gin.H{"error": "No response from model"})
				return
			}

			// Extract the content and tool calls from the response
			content, thinking, parsedToolCalls := parseChoices(response.Choices)

			// Get finish reason, default to "stop" if not provided
			finishReason := "stop"
			if response.Choices[0].FinishReason != "" {
				finishReason = string(response.Choices[0].FinishReason)
			}

			// Create Ollama-compatible response
			ollamaResponse := map[string]interface{}{
				"model":      fullModelName,
				"created_at": time.Now().Format(time.RFC3339),
				"message": map[string]interface{}{
					"role":       "assistant",
					"content":    content,
					"thinking":   thinking, // Optional: some parsers might not expect this in 'message'
					"tool_calls": parsedToolCalls,
				},
				"done":                 true,
				"finish_reason":        finishReason,
				"total_duration":       int64(1000000),
				"load_duration":        int64(1000000),
				"prompt_eval_count":    1,
				"prompt_eval_duration": int64(1000000),
				"eval_count":           1,
				"eval_duration":        int64(1000000),
			}
			c.Header("Content-Type", "application/json")
			c.JSON(http.StatusOK, ollamaResponse)
		} else {
			req := openai.ChatCompletionRequest{
				Model:    request.Model,
				Messages: openAiMessages,
				Tools:    request.Tools, // the doc (https://ollama.readthedocs.io/en/api/) says that streaming is not supported with tools, but HASS does it anyway
				Stream:   true,
			}

			//reqJson, _ := json.Marshal(req)
			//slog.Info("Request", "Request", string(reqJson))

			// Call ChatStream to get the stream
			stream, err := provider.ChatStream(c, req)
			if err != nil {
				slog.Error("Failed to create stream", "Error", err)
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}
			defer stream.Close() // Ensure stream closure

			// Set headers correctly for Newline Delimited JSON
			c.Writer.Header().Set("Content-Type", "application/x-ndjson")
			c.Writer.Header().Set("Cache-Control", "no-cache")
			c.Writer.Header().Set("Connection", "keep-alive")
			// Transfer-Encoding: chunked is set automatically by Gin

			w := c.Writer // Get the ResponseWriter
			flusher, ok := w.(http.Flusher)
			if !ok {
				slog.Error("Expected http.ResponseWriter to be an http.Flusher")
				// Sending an error to the client is difficult as headers may have already been sent
				return
			}

			var lastFinishReason string
			var toolName string
			var argsBuffer bytes.Buffer // arguments for tool calls are streamed

			flushToolCall := func() {
				if toolName == "" {
					return
				}

				var parsedToolCalls []map[string]interface{}

				// Parse arguments using YAML to be more foregiving with improper JSON
				var argsMap map[string]interface{}
				if err := yaml.Unmarshal(argsBuffer.Bytes(), &argsMap); err == nil {
					parsedToolCall := map[string]interface{}{
						"function": map[string]interface{}{
							"name":      toolName,
							"arguments": argsMap,
						},
					}
					parsedToolCalls = append(parsedToolCalls, parsedToolCall)
				} else {
					slog.Error("Failed to parse arguments for tool call", "Error", err)
				}

				toolName = ""
				argsBuffer.Reset()

				if len(parsedToolCalls) > 0 {
					// Build JSON response structure for intermediate chunks (Ollama chat format)
					responseJSON := map[string]interface{}{
						"model":      fullModelName,
						"created_at": time.Now().Format(time.RFC3339),
						"message": map[string]interface{}{
							"role":       "assistant",
							"tool_calls": parsedToolCalls,
						},
						"done": false, // Always false for intermediate chunks
					}

					// Marshal JSON
					jsonData, err := json.Marshal(responseJSON)
					if err != nil {
						slog.Error("Error marshaling intermediate response JSON", "Error", err)
						return // Return, as we cannot send data
					}
					//slog.Info("Response Chunk", "Data:", jsonData)

					// Send JSON object followed by a newline
					fmt.Fprintf(w, "%s\n", string(jsonData))
				}
			}

			// Stream responses back to the client
			for {
				response, err := stream.Recv()
				if errors.Is(err, io.EOF) {
					// End of stream from the backend provider
					break
				}
				if err != nil {
					slog.Error("Backend stream error", "Error", err)
					// Attempt to send an error in NDJSON format
					// Ollama usually just drops the connection or sends a 500 error before that
					errorMsg := map[string]string{"error": "Stream error: " + err.Error()}
					errorJson, _ := json.Marshal(errorMsg)
					fmt.Fprintf(w, "%s\n", string(errorJson)) // Send the error + \n
					flusher.Flush()
					return
				}

				if len(response.Choices) == 0 {
					continue
				}

				//slog.Info("Response", "Choices", response.Choices)

				// Extract the content and tool calls from the response
				content := ""
				thinking := ""

				if len(response.Choices) > 0 {
					delta := response.Choices[0].Delta

					toolCalls := delta.ToolCalls
					if len(toolCalls) > 0 {
						for _, tc := range toolCalls {
							//slog.Info("Tool Call", "Name", tc.Function.Name, "Arguments", tc.Function.Arguments)

							if tc.Function.Name != "" {
								flushToolCall()

								// only given in the first chunk
								toolName = tc.Function.Name
							}

							argsBuffer.WriteString(tc.Function.Arguments)
						}
					}

					content = delta.Content
					thinking = delta.ReasoningContent
					if content != "" || thinking != "" {
						flushToolCall()
					}
				}

				// Save the stop reason if present in the chunk
				if response.Choices[0].FinishReason != "" {
					lastFinishReason = string(response.Choices[0].FinishReason)
				}

				if content != "" || thinking != "" {
					// Build JSON response structure for intermediate chunks (Ollama chat format)
					responseJSON := map[string]interface{}{
						"model":      fullModelName,
						"created_at": time.Now().Format(time.RFC3339),
						"message": map[string]interface{}{
							"role":     "assistant",
							"content":  content,
							"thinking": thinking,
						},
						"done": false, // Always false for intermediate chunks
					}

					// Marshal JSON
					jsonData, err := json.Marshal(responseJSON)
					if err != nil {
						slog.Error("Error marshaling intermediate response JSON", "Error", err)
						return // Return, as we cannot send data
					}
					//slog.Info("Response Chunk", "Data:", jsonData)

					// Send JSON object followed by a newline
					fmt.Fprintf(w, "%s\n", string(jsonData))
				}

				// Flush data to send it immediately
				flusher.Flush()
			}

			// --- Sending final message (done: true) in Ollama style ---

			// Determine the stop reason (if the backend did not provide one, use 'stop')
			// Ollama uses 'stop', 'length', 'content_filter', 'tool_calls'
			if lastFinishReason == "" {
				lastFinishReason = "stop"
			}

			flushToolCall()

			// IMPORTANT: Replace nil with 0 for numeric stats fields
			finalResponse := map[string]interface{}{
				"model":      fullModelName,
				"created_at": time.Now().Format(time.RFC3339),
				"message": map[string]interface{}{
					"role":    "assistant",
					"content": "",
				},
				"done":              true,
				"done_reason":       "stop",           // Some clients look for this specific key
				"finish_reason":     lastFinishReason, // Not required for /api/chat Ollama, but does no harm
				"total_duration":    int64(1000000),
				"prompt_eval_count": 1,
				"eval_count":        1,
				"load_duration":     0,
				"eval_duration":     0,
			}

			finalJsonData, err := json.Marshal(finalResponse)
			if err != nil {
				slog.Error("Error marshaling final response JSON", "Error", err)
				return
			}

			// Send the final JSON object + newline
			fmt.Fprintf(w, "%s\n", string(finalJsonData))
			flusher.Flush()

			// IMPORTANT: For NDJSON there is NO 'data: [DONE]' marker.
			// The client detects the end of the stream by receiving an object with "done": true
			// and/or by the server closing the connection (Gin will close it automatically after exiting the handler).
		}
	})

	r.Run(":11434")
}
