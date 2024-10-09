package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"math"
	"net"
	"net/http"
	"os"
	"time"

	"github.com/CyCoreSystems/audiosocket"
	"github.com/gorilla/websocket"
	"github.com/maxhawkins/go-webrtcvad"
	"github.com/pkg/errors"
)

// MaxCallDuration is the maximum amount of time to allow a call to be up before it is terminated.
const MaxCallDuration = 2 * time.Minute

const listenAddr = ":9092"

// var dbURL = os.Getenv("DATABASE_URL")
var whisperURL = os.Getenv("WHISPER_URL")
var xttsURL = os.Getenv("XTTS_URL")
var ollamaURL = os.Getenv("OLLAMA_URL")

// slinChunkSize is the number of bytes which should be sent per Slin
// audiosocket message.  Larger data will be chunked into this size for
// transmission of the AudioSocket.
//
// This is based on 8kHz, 20ms, 16-bit signed linear.
//const slinChunkSize = 320 // 8000Hz * 20ms * 2 bytes

func init() {
}

// ErrHangup indicates that the call should be terminated or has been terminated

// ErrHangup indicates that the call should be terminated or has been terminated
var ErrHangup = errors.New("Hangup")
var client = NewOllamaClient(ollamaURL)

func main() {
	var err error
	ctx := context.Background()

	client.AddSystemPrompt("You speak only on russian and you have a limit in two-three sentences and 180 symbols!")

	log.Println("listening for AudioSocket connections on", listenAddr)
	if err = Listen(ctx); err != nil {
		log.Fatalln("listen failure:", err)
	}
	log.Println("exiting")
}

func Listen(ctx context.Context) error {
	l, err := net.Listen("tcp", listenAddr)
	if err != nil {
		return errors.Wrapf(err, "failed to bind listener to socket %s", listenAddr)
	}

	for {
		conn, err := l.Accept()
		if err != nil {
			log.Println("failed to accept new connection:", err)
			continue
		}

		go Handle(ctx, conn)
	}
}

func Handle(pCtx context.Context, c net.Conn) {
	ctx, cancel := context.WithTimeout(pCtx, MaxCallDuration)
	defer cancel()
	defer c.Close()

	vad, err := webrtcvad.New()
	if err != nil {
		log.Fatal(err)
	}

	if err := vad.SetMode(3); err != nil {
		log.Fatal(err)
	}

	id, err := audiosocket.GetID(c)
	if err != nil {
		log.Println("failed to get call ID:", err)
		return
	}
	log.Printf("processing call %s", id.String())

	rate := 16000
	silenceThreshold := 5
	var inputAudioBuffer [][]float32
	var silenceCount int

	for ctx.Err() == nil {
		m, err := audiosocket.NextMessage(c)
		if errors.Cause(err) == io.EOF {
			log.Println("audiosocket closed")
			return
		}

		switch m.Kind() {
		case audiosocket.KindHangup:
			log.Println("audiosocket received hangup command")
			return
		case audiosocket.KindError:
			log.Println("error from audiosocket")
		case audiosocket.KindSlin:
			if m.ContentLength() < 1 {
				log.Println("no audio data")
				continue
			}
			audioData := m.Payload()
			//	threshold := int16(0x02)
			//	audioDataReduced := NoiseGate(audioData, threshold)
			floatArray, err := pcmToFloat32Array(audioData)
			if err != nil {
				log.Println("error converting pcm to float32:", err)
				continue
			}

			if active, err := vad.Process(rate, audioData); err != nil {
				log.Println("Error processing VAD:", err)
			} else if active {
				inputAudioBuffer = append(inputAudioBuffer, floatArray)
				calculateAudioLength(inputAudioBuffer, rate)
				silenceCount = 0
			} else {
				silenceCount++
				if silenceCount > silenceThreshold {
					if len(inputAudioBuffer) > 0 {
						log.Println("Processing complete sentence")
						handleInputAudio(c, inputAudioBuffer)
						inputAudioBuffer = nil // Reset buffer
					}
				}
			}

		}
	}
}
func handleInputAudio(conn net.Conn, buffer [][]float32) {
	// Merge and process buffer, then send to server
	var mergedBuffer []float32
	for _, data := range buffer {
		mergedBuffer = append(mergedBuffer, data...)
	}
	length := calculateAudioLength(buffer, 16000)
	log.Println("Audio length:", length)
	if length < 0.40 {
		log.Println("Audio length is less than 0.45 seconds, skipping processing.")
		return
	}
	transcription, err := sendFloat32ArrayToServer(whisperURL, mergedBuffer)
	if err != nil {
		log.Println("Error sending data to server:", err)
		return
	}
	excludedWords := []string{"Продолжение следует...", "Субтитры сделал DimaTorzok", "Субтитры создавал DimaTorzok"}
	for _, word := range excludedWords {
		if transcription == word {
			log.Println("Transcription contains excluded word, stopping further processing.")
			return
		}
	}
	res, errO := client.GenerateNoStream("gemma2:9b", transcription, nil, "")

	if errO != nil {
		fmt.Println("Error:", err)
	}
	log.Println("Received result:", res["response"])
	data := map[string]interface{}{
		"message":  res["response"],
		"language": "ru",
		"speed":    1.0,
	}
	log.Println("Using transcription:", transcription)
	websocketSendReceive(xttsURL, data, conn)

}

func calculateAudioLength(inputAudioBuffer [][]float32, sampleRate int) float64 {
	// Calculate total number of samples in the buffer
	totalSamples := 0
	for _, buffer := range inputAudioBuffer {
		totalSamples += len(buffer)
	}

	// Calculate length in seconds
	lengthInSeconds := float64(totalSamples) / float64(sampleRate)
	return lengthInSeconds
}

func NoiseGate(input []byte, threshold int16) []byte {
	sampleCount := len(input) / 2
	output := make([]byte, len(input))

	for i := 0; i < sampleCount; i++ {
		// Extract the sample (int16) from the byte slice
		sample := int16(binary.LittleEndian.Uint16(input[i*2 : i*2+2]))

		// Apply the noise gate
		if math.Abs(float64(sample)) < float64(threshold) {
			sample = 0
		}

		// Store the processed sample back as bytes
		binary.LittleEndian.PutUint16(output[i*2:i*2+2], uint16(sample))
	}

	return output
}

func pcmToFloat32Array(pcmData []byte) ([]float32, error) {
	if len(pcmData)%2 != 0 {
		return nil, fmt.Errorf("pcm data length must be even")
	}

	float32Array := make([]float32, len(pcmData)/2)
	buf := bytes.NewReader(pcmData)

	for i := 0; i < len(float32Array); i++ {
		var sample int16
		if err := binary.Read(buf, binary.LittleEndian, &sample); err != nil {
			return nil, fmt.Errorf("failed to read sample: %v", err)
		}
		float32Array[i] = float32(sample) / 32768.0
	}

	return float32Array, nil
}

func sendFloat32ArrayToServer(serverAddress string, float32Array []float32) (string, error) {
	var buf bytes.Buffer

	for _, f := range float32Array {
		if err := binary.Write(&buf, binary.LittleEndian, f); err != nil {
			return "", fmt.Errorf("failed to write float32: %v", err)
		}
	}

	req, err := http.NewRequest("POST", serverAddress, &buf)
	if err != nil {
		return "", fmt.Errorf("error creating request: %v", err)
	}
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("error sending request: %v", err)
	}
	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("error reading response body: %v", err)
	}

	log.Println("Server response:", string(body))
	//returh response ["transcription": message]
	var result map[string]interface{}
	if err := json.Unmarshal(body, &result); err != nil {
		return "", fmt.Errorf("error unmarshalling response body: %v", err)
	}

	if transcription, ok := result["transcription"].(string); ok {
		log.Println("Transcription:", transcription)
		return transcription, nil
		//return transcription
	} else {
		return "", fmt.Errorf("transcription not found in response")
	}
}

func websocketSendReceive(uri string, data map[string]interface{}, conn net.Conn) {
	wsConn, _, err := websocket.DefaultDialer.Dial(uri, nil)
	if err != nil {
		log.Println("Failed to connect to WebSocket:", err)
		return
	}
	defer wsConn.Close()

	err = wsConn.WriteJSON(data)
	if err != nil {
		log.Println("Failed to send JSON:", err)
		return
	}

	for {
		_, message, err := wsConn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				log.Printf("Unexpected WebSocket closure: %v", err)
			}
			break
		}

		var jsonMessage map[string]interface{}
		if err := json.Unmarshal(message, &jsonMessage); err == nil {
			if typeField, ok := jsonMessage["type"].(string); ok && typeField == "end_of_audio" {
				log.Println("End of conversation")
				break
			}
			log.Println("Received message:", jsonMessage)
		} else {
			// Assume non-JSON data can be audio bytes

			if _, err := conn.Write(audiosocket.SlinMessage(message)); err != nil {
				log.Println("Error writing to connection:", err)
				break
			}
		}
	}
}

// func noiseGate(samples []float64, threshold float64) []float64 {
// 	for i, sample := range samples {
// 		if math.Abs(sample) < threshold {
// 			samples[i] = 0
// 		}
// 	}
// 	return samples
// }
