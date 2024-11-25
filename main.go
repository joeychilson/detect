package main

import (
	"context"
	"encoding/json"
	"image"
	"log"
	"net/http"
	"os"
	"strconv"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	"github.com/go-chi/cors"
	"github.com/joeychilson/onnx"

	"github.com/joeychilson/detect/yolo"
)

func handleHealth() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		JSON(w, http.StatusOK, map[string]string{"status": "ok"})
	}
}

func handleDetect(detector *yolo.Detector) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		r.Body = http.MaxBytesReader(w, r.Body, 10*1024*1024)

		confidence := float32(0.5)
		iou := float32(0.5)

		if confStr := r.URL.Query().Get("confidence"); confStr != "" {
			if conf, err := strconv.ParseFloat(confStr, 32); err == nil {
				confidence = float32(conf)
			}
		}

		if iouStr := r.URL.Query().Get("iou"); iouStr != "" {
			if iouVal, err := strconv.ParseFloat(iouStr, 32); err == nil {
				iou = float32(iouVal)
			}
		}

		if err := r.ParseMultipartForm(10 << 20); err != nil {
			Error(w, http.StatusBadRequest, "file too large")
			return
		}

		file, header, err := r.FormFile("image")
		if err != nil {
			Error(w, http.StatusBadRequest, "no file uploaded")
			return
		}
		defer file.Close()

		if !isValidFileType(header.Header.Get("Content-Type")) {
			Error(w, http.StatusBadRequest, "invalid file type")
			return
		}

		img, _, err := image.Decode(file)
		if err != nil {
			Error(w, http.StatusBadRequest, "invalid image")
			return
		}

		result, err := detector.Detect(img,
			yolo.WithConfidenceThreshold(confidence),
			yolo.WithIoUThreshold(iou),
		)
		if err != nil {
			log.Printf("detection failed: %v", err)
			Error(w, http.StatusInternalServerError, "detection failed")
			return
		}
		JSON(w, http.StatusOK, result)
	}
}

func isValidFileType(contentType string) bool {
	validTypes := map[string]bool{
		"image/jpeg": true,
		"image/png":  true,
	}
	return validTypes[contentType]
}

func JSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(data); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func Error(w http.ResponseWriter, status int, message string) {
	JSON(w, status, map[string]string{"error": message})
}

func main() {
	ctx := context.Background()

	cachePath := os.Getenv("ONNX_CACHE_PATH")

	runtime, err := onnx.New(ctx, onnx.WithCachePath(cachePath))
	if err != nil {
		log.Fatalf("failed to initialize ONNX Runtime: %v", err)
		os.Exit(1)
	}
	defer runtime.Close()

	detector, err := yolo.NewDetector("models/yolo11x.onnx")
	if err != nil {
		log.Fatalf("failed to initialize YOLO detector: %v", err)
	}
	defer detector.Close()

	router := chi.NewRouter()

	router.Use(middleware.RequestID)
	router.Use(middleware.RealIP)
	router.Use(middleware.Logger)
	router.Use(middleware.Recoverer)
	router.Use(middleware.Timeout(30 * time.Second))
	router.Use(cors.Handler(cors.Options{
		AllowedOrigins:   []string{"*"},
		AllowedMethods:   []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowedHeaders:   []string{"Accept", "Authorization", "Content-Type", "X-CSRF-Token"},
		ExposedHeaders:   []string{"Link"},
		AllowCredentials: false,
		MaxAge:           300,
	}))

	fs := http.FileServer(http.Dir("static"))
	router.Handle("/*", http.StripPrefix("/", fs))

	router.Route("/api", func(r chi.Router) {
		r.Get("/health", handleHealth())
		r.Post("/detect", handleDetect(detector))
	})

	httpServer := &http.Server{Addr: ":8080", Handler: router}

	log.Println("server starting on http://localhost:8080")
	if err := httpServer.ListenAndServe(); err != nil {
		log.Fatalf("server failed: %v", err)
	}
}
