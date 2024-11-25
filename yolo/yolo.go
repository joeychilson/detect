package yolo

import (
	"bytes"
	"errors"
	"fmt"
	"image"
	"image/color"
	"math"
	"sort"
	"strings"
	"sync"
	"text/tabwriter"
	"time"

	_ "image/jpeg"
	_ "image/png"

	ort "github.com/yalue/onnxruntime_go"
	"golang.org/x/image/draw"
	"golang.org/x/image/font"
	"golang.org/x/image/font/basicfont"
	"golang.org/x/image/math/fixed"
)

// ProviderType represents different execution providers
type ProviderType int

const (
	ProviderCPU ProviderType = iota
	ProviderCUDA
	ProviderCoreML
	ProviderDirectML
	ProviderTensorRT
)

// ProviderOptions holds configuration for different execution providers
type ProviderOptions struct {
	Provider ProviderType

	// CUDA specific options
	CUDADeviceID   int
	CUDAArenaSize  int64
	CUDAStreamSize int64

	// CoreML specific options
	CoreMLFlags uint32

	// TensorRT specific options
	TensorRTDeviceID  int
	TensorRTWorkspace int64
	TensorRTOptLevel  int

	// DirectML specific options
	DirectMLDeviceID int
}

// DefaultProviderOptions returns ProviderOptions with sensible defaults
func DefaultProviderOptions() *ProviderOptions {
	return &ProviderOptions{
		Provider:          ProviderCPU,
		CUDADeviceID:      0,
		CUDAArenaSize:     1 << 30, // 1GB
		CUDAStreamSize:    1 << 20, // 1MB
		CoreMLFlags:       0,       // Default flags
		TensorRTDeviceID:  0,
		TensorRTWorkspace: 1 << 30, // 1GB
		TensorRTOptLevel:  3,       // Maximum optimization
		DirectMLDeviceID:  0,
	}
}

// Box represents a bounding box with normalized coordinates
type Box struct {
	X1 float32 `json:"x1"` // Top-left corner X coordinate
	Y1 float32 `json:"y1"` // Top-left corner Y coordinate
	X2 float32 `json:"x2"` // Bottom-right corner X coordinate
	Y2 float32 `json:"y2"` // Bottom-right corner Y coordinate
}

// Detection represents a single object detection result
type Detection struct {
	Class      int     `json:"class"`      // Class index
	Label      string  `json:"label"`      // Class label
	Confidence float32 `json:"confidence"` // Confidence score
	Box        Box     `json:"box"`        // Bounding box
}

// DrawBox draws the bounding box on the image
func (det Detection) DrawBox(img *image.RGBA, color color.RGBA) {
	for x := int(det.Box.X1); x <= int(det.Box.X2); x++ {
		img.Set(x, int(det.Box.Y1), color)
		img.Set(x, int(det.Box.Y2), color)
	}
	for y := int(det.Box.Y1); y <= int(det.Box.Y2); y++ {
		img.Set(int(det.Box.X1), y, color)
		img.Set(int(det.Box.X2), y, color)
	}

	label := fmt.Sprintf("%s (%.0f%%)", det.Label, det.Confidence*100)
	point := fixed.Point26_6{
		X: fixed.I(int(det.Box.X1)),
		Y: fixed.I(int(det.Box.Y1 - 5)),
	}

	d := &font.Drawer{
		Dst:  img,
		Src:  image.NewUniform(color),
		Face: basicfont.Face7x13,
		Dot:  point,
	}
	d.DrawString(label)
}

// DetectionResult represents a single object detection result
type DetectionResult struct {
	Detections      []Detection   `json:"detections"`       // Detected objects
	PreprocessTime  time.Duration `json:"preprocess_time"`  // Preprocessing time
	InferenceTime   time.Duration `json:"inference_time"`   // Inference time
	PostprocessTime time.Duration `json:"postprocess_time"` // Postprocessing time
	DetectionTime   time.Duration `json:"detection_time"`   // Total detection time
}

// String returns a human-readable summary of the detection results
func (r *DetectionResult) String() string {
	var buf bytes.Buffer
	w := tabwriter.NewWriter(&buf, 0, 0, 2, ' ', 0)

	fmt.Fprintf(w, "Detection Results Summary:\n")
	fmt.Fprintf(w, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
	fmt.Fprintf(w, "Timing:\n")
	fmt.Fprintf(w, "  Preprocess:\t%v\n", r.PreprocessTime)
	fmt.Fprintf(w, "  Inference:\t%v\n", r.InferenceTime)
	fmt.Fprintf(w, "  Postprocess:\t%v\n", r.PostprocessTime)
	fmt.Fprintf(w, "  Total Time:\t%v\n", r.DetectionTime)
	fmt.Fprintf(w, "\n")

	classGroups := make(map[string][]Detection)
	for _, d := range r.Detections {
		classGroups[d.Label] = append(classGroups[d.Label], d)
	}

	if len(r.Detections) > 0 {
		fmt.Fprintf(w, "Detections (%d total):\n", len(r.Detections))
		fmt.Fprintf(w, "  Class\tCount\tAvg Confidence\tConfidence Range\n")
		fmt.Fprintf(w, "  ─────\t─────\t──────────────\t────────────────\n")

		var classes []string
		for class := range classGroups {
			classes = append(classes, class)
		}
		sort.Strings(classes)

		for _, class := range classes {
			detections := classGroups[class]

			// Calculate statistics
			var sumConf float32
			minConf := float32(1.0)
			maxConf := float32(0.0)

			for _, d := range detections {
				sumConf += d.Confidence
				if d.Confidence < minConf {
					minConf = d.Confidence
				}
				if d.Confidence > maxConf {
					maxConf = d.Confidence
				}
			}

			avgConf := sumConf / float32(len(detections))

			fmt.Fprintf(w, "  %s\t%d\t%.1f%%\t%.1f%% - %.1f%%\n",
				class,
				len(detections),
				avgConf*100,
				minConf*100,
				maxConf*100,
			)
		}
	} else {
		fmt.Fprintf(w, "No detections found\n")
	}

	if len(r.Detections) > 0 {
		fmt.Fprintf(w, "\nDetailed Detections:\n")
		fmt.Fprintf(w, "  ID\tClass\tConfidence\tBounding Box\n")
		fmt.Fprintf(w, "  ──\t─────\t──────────\t────────────\n")

		for i, d := range r.Detections {
			fmt.Fprintf(w, "  %d\t%s\t%.1f%%\t(%.1f, %.1f, %.1f, %.1f)\n",
				i+1,
				d.Label,
				d.Confidence*100,
				d.Box.X1,
				d.Box.Y1,
				d.Box.X2,
				d.Box.Y2,
			)
		}
	}

	w.Flush()
	return strings.TrimSpace(buf.String())
}

// Detector manages the YOLO object detection model
type Detector struct {
	session      *ort.AdvancedSession
	mutex        sync.Mutex
	inputTensor  *ort.Tensor[float32]
	outputTensor *ort.Tensor[float32]
}

// DetectorOptions contains options for creating a new detector
type DetectorOptions struct {
	ModelPath      string           // Path to the ONNX model file
	InterOpThreads int              // Number of inter-op threads
	IntraOpThreads int              // Number of intra-op threads
	CpuMemArena    bool             // Enable CPU memory arena
	MemPattern     bool             // Enable memory pattern optimization
	Provider       *ProviderOptions // Execution provider options
}

// WithInterOpThreads sets the number of inter-op threads
func WithInterOpThreads(threads int) func(*DetectorOptions) {
	return func(opts *DetectorOptions) { opts.InterOpThreads = threads }
}

// WithIntraOpThreads sets the number of intra-op threads
func WithIntraOpThreads(threads int) func(*DetectorOptions) {
	return func(opts *DetectorOptions) { opts.IntraOpThreads = threads }
}

// WithCpuMemArena enables CPU memory arena
func WithCpuMemArena() func(*DetectorOptions) {
	return func(opts *DetectorOptions) { opts.CpuMemArena = true }
}

// WithMemPattern enables memory pattern optimization
func WithMemPattern() func(*DetectorOptions) {
	return func(opts *DetectorOptions) { opts.MemPattern = true }
}

// WithProvider sets the execution provider options
func WithProvider(provider *ProviderOptions) func(*DetectorOptions) {
	return func(opts *DetectorOptions) { opts.Provider = provider }
}

// NewDetector creates a new YOLO detector instance
func NewDetector(modelPath string, opts ...func(*DetectorOptions)) (*Detector, error) {
	options := &DetectorOptions{
		ModelPath:      modelPath,
		InterOpThreads: 1,
		IntraOpThreads: 4,
		CpuMemArena:    true,
		MemPattern:     true,
		Provider:       DefaultProviderOptions(),
	}
	for _, opt := range opts {
		opt(options)
	}

	sessionOptions, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("failed to create session options: %w", err)
	}
	defer sessionOptions.Destroy()

	if err := sessionOptions.SetInterOpNumThreads(options.InterOpThreads); err != nil {
		return nil, fmt.Errorf("failed to set inter op threads: %w", err)
	}
	if err := sessionOptions.SetIntraOpNumThreads(options.IntraOpThreads); err != nil {
		return nil, fmt.Errorf("failed to set intra op threads: %w", err)
	}

	if err := sessionOptions.SetCpuMemArena(options.CpuMemArena); err != nil {
		return nil, fmt.Errorf("failed to enable CPU memory arena: %w", err)
	}
	if err := sessionOptions.SetMemPattern(options.MemPattern); err != nil {
		return nil, fmt.Errorf("failed to enable memory pattern optimization: %w", err)
	}

	if err := configureProvider(sessionOptions, options.Provider); err != nil {
		return nil, fmt.Errorf("failed to configure provider: %w", err)
	}

	inputShape := ort.NewShape(1, 3, 640, 640)
	inputTensor, err := ort.NewEmptyTensor[float32](inputShape)
	if err != nil {
		return nil, fmt.Errorf("failed to create input tensor: %w", err)
	}

	outputShape := ort.NewShape(1, 84, 8400)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return nil, fmt.Errorf("failed to create output tensor: %w", err)
	}

	session, err := ort.NewAdvancedSession(
		modelPath,
		[]string{"images"},
		[]string{"output0"},
		[]ort.Value{inputTensor},
		[]ort.Value{outputTensor},
		sessionOptions,
	)
	if err != nil {
		inputTensor.Destroy()
		outputTensor.Destroy()
		return nil, fmt.Errorf("failed to create session: %w", err)
	}

	return &Detector{
		session:      session,
		inputTensor:  inputTensor,
		outputTensor: outputTensor,
	}, nil
}

// Close releases all resources associated with the detector
func (d *Detector) Close() error {
	d.mutex.Lock()
	defer d.mutex.Unlock()

	if err := d.inputTensor.Destroy(); err != nil {
		return fmt.Errorf("failed to destroy input tensor: %w", err)
	}
	if err := d.outputTensor.Destroy(); err != nil {
		return fmt.Errorf("failed to destroy output tensor: %w", err)
	}
	if err := d.session.Destroy(); err != nil {
		return fmt.Errorf("failed to destroy session: %w", err)
	}
	return nil
}

// DetectOptions contains options for object detection
type DetectOptions struct {
	ConfidenceThreshold float32 `json:"confidence_threshold"` // Confidence threshold for detections
	IoUThreshold        float32 `json:"iou_threshold"`        // IoU threshold for NMS
}

// WithConfidenceThreshold sets the confidence threshold for detections
func WithConfidenceThreshold(threshold float32) func(*DetectOptions) {
	return func(opts *DetectOptions) {
		opts.ConfidenceThreshold = threshold
	}
}

// WithIoUThreshold sets the IoU threshold for NMS
func WithIoUThreshold(threshold float32) func(*DetectOptions) {
	return func(opts *DetectOptions) {
		opts.IoUThreshold = threshold
	}
}

// Detect performs object detection on the provided image
func (d *Detector) Detect(img image.Image, opts ...func(*DetectOptions)) (*DetectionResult, error) {
	d.mutex.Lock()
	defer d.mutex.Unlock()

	startTime := time.Now()

	options := &DetectOptions{ConfidenceThreshold: 0.25, IoUThreshold: 0.45}
	for _, opt := range opts {
		opt(options)
	}

	// Preprocess the image
	preprocessStart := time.Now()
	if err := d.preprocess(img); err != nil {
		return nil, fmt.Errorf("failed to preprocess image: %w", err)
	}
	preprocessTime := time.Since(preprocessStart)

	// Run inference
	inferenceStart := time.Now()
	if err := d.session.Run(); err != nil {
		return nil, fmt.Errorf("failed to run inference: %w", err)
	}
	inferenceTime := time.Since(inferenceStart)

	// Process the results and return detections
	postprocessStart := time.Now()
	detections, err := d.postprocess(img, options.ConfidenceThreshold, options.IoUThreshold)
	if err != nil {
		return nil, fmt.Errorf("failed to postprocess detections: %w", err)
	}
	postprocessTime := time.Since(postprocessStart)

	return &DetectionResult{
		Detections:      detections,
		PreprocessTime:  preprocessTime,
		InferenceTime:   inferenceTime,
		PostprocessTime: postprocessTime,
		DetectionTime:   time.Since(startTime),
	}, nil
}

func (d *Detector) preprocess(img image.Image) error {
	inputData := d.inputTensor.GetData()

	if img == nil {
		return errors.New("nil image")
	}

	origSize := image.Point{X: img.Bounds().Dx(), Y: img.Bounds().Dy()}

	if origSize.X < 1 || origSize.Y < 1 {
		return errors.New("invalid image dimensions")
	}

	resized := image.NewRGBA(image.Rect(0, 0, 640, 640))
	draw.BiLinear.Scale(resized, resized.Bounds(), img, img.Bounds(), draw.Over, nil)

	bounds := resized.Bounds()
	width, height := bounds.Dx(), bounds.Dy()

	pixels := make([]float32, 3*height*width)

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r, g, b, _ := resized.At(x, y).RGBA()

			rf := float32(r>>8) / 255.0
			gf := float32(g>>8) / 255.0
			bf := float32(b>>8) / 255.0

			pixels[0*height*width+y*width+x] = rf
			pixels[1*height*width+y*width+x] = gf
			pixels[2*height*width+y*width+x] = bf
		}
	}

	for i, pixel := range pixels {
		inputData[i] = pixel
	}
	return nil
}

func (d *Detector) postprocess(img image.Image, confidenceThreshold, iouThreshold float32) ([]Detection, error) {
	outputData := d.outputTensor.GetData()

	originalWidth, originalHeight := img.Bounds().Dx(), img.Bounds().Dy()

	detections := make([]Detection, 0)

	for i := 0; i < 8400; i++ {
		var (
			maxConf  float32
			maxClass int
		)

		for j := 0; j < 80; j++ {
			conf := outputData[8400*(j+4)+i]
			if conf > maxConf {
				maxConf = conf
				maxClass = j
			}
		}

		if maxConf < confidenceThreshold {
			continue
		}

		xc, yc := outputData[i], outputData[8400+i]
		w, h := outputData[2*8400+i], outputData[3*8400+i]
		x1 := (xc - w/2) / 640 * float32(originalWidth)
		y1 := (yc - h/2) / 640 * float32(originalHeight)
		x2 := (xc + w/2) / 640 * float32(originalWidth)
		y2 := (yc + h/2) / 640 * float32(originalHeight)

		detections = append(detections, Detection{
			Class:      maxClass,
			Label:      labels[maxClass],
			Confidence: maxConf,
			Box: Box{
				X1: x1,
				Y1: y1,
				X2: x2,
				Y2: y2,
			},
		})
	}
	return d.nms(detections, iouThreshold), nil
}

func (d *Detector) nms(detections []Detection, iouThreshold float32) []Detection {
	if len(detections) == 0 {
		return detections
	}

	sort.Slice(detections, func(i, j int) bool {
		return detections[i].Confidence > detections[j].Confidence
	})

	keep := make([]bool, len(detections))
	for i := range keep {
		keep[i] = true
	}

	for i := 0; i < len(detections)-1; i++ {
		if !keep[i] {
			continue
		}
		for j := i + 1; j < len(detections); j++ {
			if !keep[j] {
				continue
			}
			if detections[i].Class != detections[j].Class {
				continue
			}

			box1 := [4]float32{detections[i].Box.X1, detections[i].Box.Y1, detections[i].Box.X2, detections[i].Box.Y2}
			box2 := [4]float32{detections[j].Box.X1, detections[j].Box.Y1, detections[j].Box.X2, detections[j].Box.Y2}

			if iou(box1, box2) > iouThreshold {
				keep[j] = false
			}
		}
	}

	var result []Detection
	for i, detection := range detections {
		if keep[i] {
			result = append(result, detection)
		}
	}
	return result
}

func iou(a, b [4]float32) float32 {
	intersectX1 := math.Max(float64(a[0]), float64(b[0]))
	intersectY1 := math.Max(float64(a[1]), float64(b[1]))
	intersectX2 := math.Min(float64(a[2]), float64(b[2]))
	intersectY2 := math.Min(float64(a[3]), float64(b[3]))

	intersectArea := math.Max(0, intersectX2-intersectX1) * math.Max(0, intersectY2-intersectY1)

	areaA := float64((a[2] - a[0]) * (a[3] - a[1]))
	areaB := float64((b[2] - b[0]) * (b[3] - b[1]))

	return float32(intersectArea / (areaA + areaB - intersectArea))
}

// configureProvider attempts to configure the specified provider
func configureProvider(sessionOptions *ort.SessionOptions, opts *ProviderOptions) error {
	switch opts.Provider {
	case ProviderCUDA:
		return configureCUDA(sessionOptions, opts)
	case ProviderCoreML:
		return configureCoreML(sessionOptions, opts)
	case ProviderDirectML:
		return configureDirectML(sessionOptions, opts)
	case ProviderTensorRT:
		return configureTensorRT(sessionOptions, opts)
	default:
		return nil
	}
}

func configureCUDA(sessionOptions *ort.SessionOptions, opts *ProviderOptions) error {
	cudaOpts, err := ort.NewCUDAProviderOptions()
	if err != nil {
		return fmt.Errorf("failed to create CUDA provider options: %w", err)
	}
	defer cudaOpts.Destroy()

	options := map[string]string{
		"device_id":                 fmt.Sprintf("%d", opts.CUDADeviceID),
		"arena_extend_strategy":     "kNextPowerOfTwo",
		"gpu_mem_limit":             fmt.Sprintf("%d", opts.CUDAArenaSize),
		"cudnn_conv_algo_search":    "EXHAUSTIVE",
		"do_copy_in_default_stream": "1",
	}

	if err := cudaOpts.Update(options); err != nil {
		return fmt.Errorf("failed to update CUDA provider options: %w", err)
	}

	if err := sessionOptions.AppendExecutionProviderCUDA(cudaOpts); err != nil {
		return fmt.Errorf("failed to append CUDA provider: %w", err)
	}
	return nil
}

func configureCoreML(sessionOptions *ort.SessionOptions, opts *ProviderOptions) error {
	if err := sessionOptions.AppendExecutionProviderCoreML(opts.CoreMLFlags); err != nil {
		return fmt.Errorf("failed to append CoreML provider: %w", err)
	}
	return nil
}

func configureDirectML(sessionOptions *ort.SessionOptions, opts *ProviderOptions) error {
	if err := sessionOptions.AppendExecutionProviderDirectML(opts.DirectMLDeviceID); err != nil {
		return fmt.Errorf("failed to append DirectML provider: %w", err)
	}
	return nil
}

func configureTensorRT(sessionOptions *ort.SessionOptions, opts *ProviderOptions) error {
	trtOpts, err := ort.NewTensorRTProviderOptions()
	if err != nil {
		return fmt.Errorf("failed to create TensorRT provider options: %w", err)
	}
	defer trtOpts.Destroy()

	options := map[string]string{
		"device_id":                    fmt.Sprintf("%d", opts.TensorRTDeviceID),
		"max_workspace_size":           fmt.Sprintf("%d", opts.TensorRTWorkspace),
		"optimization_level":           fmt.Sprintf("%d", opts.TensorRTOptLevel),
		"trt_max_partition_iterations": "1000",
		"trt_min_subgraph_size":        "1",
	}

	if err := trtOpts.Update(options); err != nil {
		return fmt.Errorf("failed to update TensorRT provider options: %w", err)
	}

	if err := sessionOptions.AppendExecutionProviderTensorRT(trtOpts); err != nil {
		return fmt.Errorf("failed to append TensorRT provider: %w", err)
	}
	return nil
}

var labels = []string{
	"person",
	"bicycle",
	"car",
	"motorcycle",
	"airplane",
	"bus",
	"train",
	"truck",
	"boat",
	"traffic light",
	"fire hydrant",
	"stop sign",
	"parking meter",
	"bench",
	"bird",
	"cat",
	"dog",
	"horse",
	"sheep",
	"cow",
	"elephant",
	"bear",
	"zebra",
	"giraffe",
	"backpack",
	"umbrella",
	"handbag",
	"tie",
	"suitcase",
	"frisbee",
	"skis",
	"snowboard",
	"sports ball",
	"kite",
	"baseball bat",
	"baseball glove",
	"skateboard",
	"surfboard",
	"tennis racket",
	"bottle",
	"wine glass",
	"cup",
	"fork",
	"knife",
	"spoon",
	"bowl",
	"banana",
	"apple",
	"sandwich",
	"orange",
	"broccoli",
	"carrot",
	"hot dog",
	"pizza",
	"donut",
	"cake",
	"chair",
	"couch",
	"potted plant",
	"bed",
	"dining table",
	"toilet",
	"tv",
	"laptop",
	"mouse",
	"remote",
	"keyboard",
	"cell phone",
	"microwave",
	"oven",
	"toaster",
	"sink",
	"refrigerator",
	"book",
	"clock",
	"vase",
	"scissors",
	"teddy bear",
	"hair drier",
	"toothbrush",
}
