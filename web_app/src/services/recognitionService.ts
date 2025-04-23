import * as tf from '@tensorflow/tfjs';

export interface RecognitionResult {
  gesture: string;
  timestamp: number;
  confidence?: number;
}

const gestureLabels = [
  "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
  "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
  "U", "V", "W", "X", "Y", "Z", "0", "1", "2", "3",
  "4", "5", "6", "7", "8", "9", "10"
];

class RecognitionService {
  private isInitialized: boolean = false;
  private isRunning: boolean = false;
  private videoElement: HTMLVideoElement | null = null;
  private canvasElement: HTMLCanvasElement | null = null;
  private callbackFn: ((result: RecognitionResult) => void) | null = null;
  private model: tf.LayersModel | null = null;
  private lastSpokenGesture: string | null = null;
  private predictionHistory: number[] = [];
  private predictionInterval: number | null = null;
  private hands: any = null;

  async initialize(videoElement: HTMLVideoElement, canvasElement: HTMLCanvasElement): Promise<boolean> {
    try {
      await tf.ready();
      console.log("TensorFlow.js initialized successfully");

      this.model = await tf.loadLayersModel('model/model.json');
      console.log("Model loaded successfully");
      console.log("Model input shape:", this.model.inputs[0].shape);

      this.videoElement = videoElement;
      this.canvasElement = canvasElement;
      this.isInitialized = true;

      if (typeof window !== 'undefined' && (window as any).Hands) {
        await this.setupHandTracking();
      } else {
        console.warn("MediaPipe Hands not available.");
      }

      console.log("Recognition service initialized");
      return true;
    } catch (err) {
      console.error("Failed to initialize:", err);
      return false;
    }
  }

  private async setupHandTracking() {
    if (typeof window === 'undefined' || !(window as any).Hands) return;

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

    console.log("Hand tracking initialized");
  }

  private processHandResults(results: any) {
    if (!this.isRunning || !this.canvasElement || !this.videoElement) return;

    const ctx = this.canvasElement.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height);

    ctx.drawImage(
      this.videoElement,
      0, 0,
      this.canvasElement.width,
      this.canvasElement.height
    );

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      const hand = results.multiHandLandmarks[0];

      // Step 1: Extract and flatten landmarks
      const flattened = hand.flatMap((pt: any) => [pt.x, pt.y, pt.z]);

      // Step 2: Normalize landmarks
      const maxVal = Math.max(...flattened.map(Math.abs)) || 1;
      const normalized = flattened.map(val => val / maxVal);

      // Step 3: Ensure correct length
      if (normalized.length > 63) {
        normalized.splice(63);
      } else if (normalized.length < 63) {
        while (normalized.length < 63) normalized.push(0);
      }

      // Step 4: Predict
      const inputTensor = tf.tensor(normalized).reshape([1, 1, 63]);
      this.predictFromCanvas(inputTensor);

    } else {
      // No hand detected
      this.predictionHistory = []; // Clear history when no hand
      if (this.callbackFn) {
        this.callbackFn({
          gesture: "-",
          timestamp: Date.now()
        });
      }
    }
  }

  private async predictFromCanvas(inputTensor: tf.Tensor) {
    if (!this.model || !this.callbackFn) return;

    try {
      const prediction = this.model.predict(inputTensor) as tf.Tensor;
      const scores = await prediction.data();
      const maxIndex = scores.indexOf(Math.max(...scores));
      const confidence = scores[maxIndex];

      inputTensor.dispose();
      prediction.dispose();

      if (confidence > 0.6) {
        this.predictionHistory.push(maxIndex);
        if (this.predictionHistory.length > 5) {
          this.predictionHistory.shift(); // Keep history of last 5
        }

        const mostCommon = this.mostFrequent(this.predictionHistory);

        const gesture = gestureLabels[mostCommon];

        const result = {
          gesture: gesture,
          timestamp: Date.now(),
          confidence: confidence
        };

        if (gesture !== this.lastSpokenGesture) {
          this.speakText(gesture);
          this.lastSpokenGesture = gesture;
        }

        this.callbackFn(result);
      } else {
        this.callbackFn({
          gesture: "?",
          timestamp: Date.now(),
          confidence: confidence
        });
      }
    } catch (error) {
      console.error("Prediction error:", error);

      if (this.callbackFn) {
        this.callbackFn({
          gesture: "!",
          timestamp: Date.now()
        });
      }
    }
  }

  private mostFrequent(array: number[]): number {
    const counts = new Map<number, number>();
    array.forEach(num => {
      counts.set(num, (counts.get(num) || 0) + 1);
    });

    let maxCount = 0;
    let mostCommon = array[0];

    counts.forEach((count, num) => {
      if (count > maxCount) {
        maxCount = count;
        mostCommon = num;
      }
    });

    return mostCommon;
  }

  async start(callback: (result: RecognitionResult) => void): Promise<void> {
    if (!this.isInitialized) {
      throw new Error("Recognition service not initialized");
    }

    this.callbackFn = callback;
    this.isRunning = true;

    if (this.hands && this.videoElement) {
      const camera = new (window as any).Camera(this.videoElement, {
        onFrame: async () => {
          if (this.hands && this.isRunning) {
            await this.hands.send({ image: this.videoElement });
          }
        },
        width: 640,
        height: 480
      });
      camera.start();
    }
  }

  stop(): void {
    this.isRunning = false;
    if (this.predictionInterval !== null) {
      window.clearInterval(this.predictionInterval);
      this.predictionInterval = null;
    }
  }

  speakText(text: string): void {
    if ('speechSynthesis' in window) {
      const synth = window.speechSynthesis;
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.2;
      utterance.pitch = 1.0;
      utterance.volume = 1.0;

      synth.cancel();
      synth.speak(utterance);
    }
  }
}

export const recognitionService = new RecognitionService();
