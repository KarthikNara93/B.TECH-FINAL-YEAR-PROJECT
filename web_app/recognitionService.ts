import * as tf from '@tensorflow/tfjs';

// Type for recognition results
export interface RecognitionResult {
  gesture: string;
  timestamp: number;
  confidence?: number;
}

// Gesture labels - matching your model's labels
const gestureLabels = ["A", "B", "C", "D", "E", "F"];

class RecognitionService {
  private isInitialized: boolean = false;
  private isRunning: boolean = false;
  private videoElement: HTMLVideoElement | null = null;
  private canvasElement: HTMLCanvasElement | null = null;
  private callbackFn: ((result: RecognitionResult) => void) | null = null;
  private model: tf.LayersModel | null = null;
  private lastSpokenGesture: string | null = null;
  private predictionInterval: number | null = null;
  private handDetector: any = null;
  private hands: any = null;
  
  // Initialize the service with video and canvas elements
  async initialize(
    videoElement: HTMLVideoElement,
    canvasElement: HTMLCanvasElement
  ): Promise<boolean> {
    try {
      await tf.ready();
      console.log("TensorFlow.js initialized successfully");
      
      // Load the TensorFlow.js model
      console.log("Loading TensorFlow.js model...");
      
      try {
        // Load the actual model
        this.model = await tf.loadLayersModel('model/model.json');
        console.log("TensorFlow model loaded successfully");
      } catch (modelError) {
        console.error("Could not load TensorFlow model:", modelError);
        
        // Create a fallback model for demonstration
        // Fix: Create a sequential model first, then assign it to this.model
        const fallbackModel = tf.sequential();
        fallbackModel.add(tf.layers.dense({units: 6, activation: 'softmax', inputShape: [64*64*3]}));
        this.model = fallbackModel;
        console.warn("Created fallback model since actual model couldn't be loaded");
      }
      
      this.videoElement = videoElement;
      this.canvasElement = canvasElement;
      this.isInitialized = true;
      
      // Initialize MediaPipe Hands if available in the global scope
      try {
        if (typeof window !== 'undefined' && (window as any).Hands) {
          await this.setupHandTracking();
        } else {
          console.warn("MediaPipe Hands not available. Will use basic frame analysis.");
        }
      } catch (handError) {
        console.warn("Could not initialize hand tracking:", handError);
      }
      
      console.log("Recognition service initialized");
      return true;
    } catch (err) {
      console.error("Failed to initialize recognition service:", err);
      return false;
    }
  }
  
  // Setup MediaPipe hand tracking
  private async setupHandTracking() {
    if (typeof window === 'undefined' || !(window as any).Hands) return;
    
    try {
      const Hands = (window as any).Hands;
      this.hands = new Hands({
        locateFile: (file: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
      });
      
      this.hands.setOptions({
        maxNumHands: 1,
        modelComplexity: 1,
        minDetectionConfidence: 0.7,
        minTrackingConfidence: 0.7
      });
      
      this.hands.onResults((results: any) => {
        this.processHandResults(results);
      });
      
      console.log("Hand tracking initialized successfully");
    } catch (error) {
      console.error("Error setting up hand tracking:", error);
    }
  }
  
  // Process hand tracking results
  private processHandResults(results: any) {
    if (!this.isRunning || !this.canvasElement || !this.videoElement) return;
    
    const ctx = this.canvasElement.getContext('2d');
    if (!ctx) return;
    
    // Clear the canvas
    ctx.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height);
    
    // Draw the current frame from video
    ctx.drawImage(
      this.videoElement,
      0, 0,
      this.canvasElement.width,
      this.canvasElement.height
    );
    
    // If hands are detected
    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      const hand = results.multiHandLandmarks[0];
      
      // Get bounding box
      const xs = hand.map((pt: any) => pt.x * this.canvasElement!.width);
      const ys = hand.map((pt: any) => pt.y * this.canvasElement!.height);
      const x0 = Math.max(Math.min(...xs) - 20, 0);
      const y0 = Math.max(Math.min(...ys) - 20, 0);
      const x1 = Math.min(Math.max(...xs) + 20, this.canvasElement!.width);
      const y1 = Math.min(Math.max(...ys) + 20, this.canvasElement!.height);
      
      // Create a temporary canvas for the cropped hand
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = 64;
      tempCanvas.height = 64;
      const tempCtx = tempCanvas.getContext('2d');
      
      if (tempCtx) {
        // Draw the cropped hand to the temporary canvas
        tempCtx.drawImage(
          this.canvasElement,
          x0, y0, x1 - x0, y1 - y0,
          0, 0, 64, 64
        );
        
        // Make prediction with the cropped hand
        this.predictFromCanvas(tempCanvas);
      }
    } else {
      // No hands detected
      if (this.callbackFn) {
        this.callbackFn({
          gesture: "-",
          timestamp: Date.now()
        });
      }
    }
  }
  
  // Make prediction from a canvas
  private async predictFromCanvas(canvas: HTMLCanvasElement) {
    if (!this.model || !this.callbackFn) return;
    
    try {
      // Preprocess the canvas
      const inputTensor = tf.browser.fromPixels(canvas)
        .resizeBilinear([64, 64])
        .toFloat()
        .div(255.0);
      
      // Check the model's input shape to determine if reshape is needed
      const inputShape = this.model.inputs[0].shape;
      console.log("Model expects input shape:", inputShape);
      
      let reshapedInput;
      // For a model expecting 5D input [batch, time, height, width, channels]
      if (inputShape.length === 5) {
        // Add time dimension and batch dimension
        // Assuming model expects [batch_size, time_steps, height, width, channels]
        const timeSteps = inputShape[1] || 1; // Get expected time steps or default to 1
        reshapedInput = inputTensor.reshape([1, timeSteps, 64, 64, 3]);
      } 
      // For CNN expecting 4D input [batch, height, width, channels]
      else {
        reshapedInput = inputTensor.expandDims(0);
      }
      
      // Make prediction
      const prediction = this.model.predict(reshapedInput) as tf.Tensor;
      const scores = await prediction.data();
      const maxIndex = scores.indexOf(Math.max(...scores));
      const confidence = scores[maxIndex];
      
      // Clean up tensors
      inputTensor.dispose();
      reshapedInput.dispose();
      prediction.dispose();
      
      // Only report if confidence is high enough
      if (confidence > 0.6) {
        const gesture = gestureLabels[maxIndex];
        
        const result = {
          gesture: gesture,
          timestamp: Date.now(),
          confidence: confidence
        };
        
        // Only speak when gesture changes
        if (gesture !== this.lastSpokenGesture) {
          this.speakText(gesture);
          this.lastSpokenGesture = gesture;
        }
        
        this.callbackFn(result);
      } else {
        // Not confident enough
        this.callbackFn({
          gesture: "?",
          timestamp: Date.now(),
          confidence: confidence
        });
      }
    } catch (error) {
      console.error("Error predicting from canvas:", error);
      
      // Log additional information about the error for debugging
      if (error instanceof Error) {
        console.error("Error details:", error.message);
      }
      
      // Notify the UI of the error
      if (this.callbackFn) {
        this.callbackFn({
          gesture: "!",
          timestamp: Date.now()
        });
      }
    }
  }
  
  // Start the recognition process
  async start(callback: (result: RecognitionResult) => void): Promise<void> {
    if (!this.isInitialized) {
      throw new Error("Recognition service not initialized");
    }
    
    this.callbackFn = callback;
    this.isRunning = true;
    
    // If we have MediaPipe hands, use the camera
    if (this.hands && this.videoElement) {
      try {
        const camera = new (window as any).Camera(this.videoElement, {
          onFrame: async () => {
            if (this.hands && this.isRunning) {
              await this.hands.send({image: this.videoElement});
            }
          },
          width: 640,
          height: 480
        });
        camera.start();
      } catch (error) {
        console.error("Error starting camera:", error);
        this.startFallbackPrediction();
      }
    } else {
      // Fallback to regular frame analysis
      this.startFallbackPrediction();
    }
  }
  
  // Start fallback prediction without hand tracking
  private startFallbackPrediction() {
    // Clear any existing interval
    if (this.predictionInterval !== null) {
      window.clearInterval(this.predictionInterval);
    }
    
    // Use an interval to analyze frames
    this.predictionInterval = window.setInterval(() => {
      if (this.isRunning && this.videoElement && this.canvasElement) {
        const ctx = this.canvasElement.getContext('2d');
        if (ctx) {
          // Draw video frame to canvas
          ctx.drawImage(
            this.videoElement,
            0, 0,
            this.canvasElement.width,
            this.canvasElement.height
          );
          
          // Make prediction from the full frame
          this.predictFromCanvas(this.canvasElement);
        }
      }
    }, 500);
  }
  
  // Stop the recognition process
  stop(): void {
    this.isRunning = false;
    if (this.predictionInterval !== null) {
      window.clearInterval(this.predictionInterval);
      this.predictionInterval = null;
    }
  }
  
  // Draw the current video frame to the canvas WITHOUT hand landmarks visualization
  private drawVideoFrame(): void {
    if (!this.videoElement || !this.canvasElement) return;
    
    const ctx = this.canvasElement.getContext('2d');
    if (!ctx) return;
    
    // Match canvas size to video
    this.canvasElement.width = this.videoElement.videoWidth;
    this.canvasElement.height = this.videoElement.videoHeight;
    
    // Draw the video frame only - NO hand landmarks
    ctx.drawImage(
      this.videoElement,
      0, 0,
      this.canvasElement.width,
      this.canvasElement.height
    );
  }
  
  // Helper method to speak text
  speakText(text: string): void {
    if ('speechSynthesis' in window) {
      const synth = window.speechSynthesis;
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.2;
      utterance.pitch = 1.0;
      utterance.volume = 1.0;
      
      synth.cancel(); // Prevents queuing delays
      synth.speak(utterance);
    }
  }
}

// Create and export a singleton instance
export const recognitionService = new RecognitionService();