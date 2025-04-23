
import { useState, useRef, useEffect } from 'react';
import { Play, Pause, AlertCircle, Camera } from 'lucide-react';
import { recognitionService, RecognitionResult } from '../services/recognitionService';
import ResultDisplay from './ResultDisplay';

const RecognitionView = () => {
  const [isInitialized, setIsInitialized] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [isCameraLoaded, setIsCameraLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [latestResult, setLatestResult] = useState<RecognitionResult | null>(null);
  const [recentResults, setRecentResults] = useState<RecognitionResult[]>([]);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  // Load MediaPipe scripts
  useEffect(() => {
    const loadScripts = async () => {
      try {
        // Load MediaPipe Hands library if not already loaded
        if (!(window as any).Hands) {
          const script1 = document.createElement('script');
          script1.src = "https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js";
          script1.async = true;
          document.body.appendChild(script1);
          
          // Wait for the script to load
          await new Promise((resolve) => {
            script1.onload = resolve;
            // Timeout after 5 seconds
            setTimeout(resolve, 5000);
          });
        }
        
        // Load Camera utils
        if (!(window as any).Camera) {
          const script2 = document.createElement('script');
          script2.src = "https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js";
          script2.async = true;
          document.body.appendChild(script2);
          
          // Wait for the script to load
          await new Promise((resolve) => {
            script2.onload = resolve;
            // Timeout after 5 seconds
            setTimeout(resolve, 5000);
          });
        }
      } catch (err) {
        console.error("Error loading MediaPipe scripts:", err);
      }
    };
    
    loadScripts();
  }, []);
  
  // Initialize camera and recognition service
  useEffect(() => {
    const initializeCamera = async () => {
      try {
        if (!videoRef.current) return;
        
        const stream = await navigator.mediaDevices.getUserMedia({ 
          video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
            facingMode: 'user'
          }
        });
        
        videoRef.current.srcObject = stream;
        setIsCameraLoaded(true);
        
        // Initialize recognition service after camera is loaded
        videoRef.current.onloadedmetadata = async () => {
          if (videoRef.current && canvasRef.current) {
            const initialized = await recognitionService.initialize(
              videoRef.current, 
              canvasRef.current
            );
            setIsInitialized(initialized);
          }
        };
      } catch (err) {
        setError('Unable to access camera. Please ensure you have granted camera permissions.');
        console.error('Camera access error:', err);
        
        // If camera fails, we'll still initialize the service for demo purposes
        if (canvasRef.current && videoRef.current) {
          const initialized = await recognitionService.initialize(
            videoRef.current,
            canvasRef.current
          );
          setIsInitialized(initialized);
          setIsCameraLoaded(true); // Pretend camera is loaded for demo
        }
      }
    };
    
    // Give time for MediaPipe scripts to load
    const timer = setTimeout(() => {
      initializeCamera();
    }, 1000);
    
    // Cleanup on unmount
    return () => {
      clearTimeout(timer);
      recognitionService.stop();
      
      // Stop and release camera
      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, []);
  
  // Handle recognition results
  const handleRecognitionResult = (result: RecognitionResult) => {
    if (result.gesture === "-" || result.gesture === "?") {
      // Don't update the UI for no gesture or low confidence
      return;
    }
    
    setLatestResult(result);
    
    // Add to recent results and keep only the last 5
    setRecentResults(prev => {
      const updated = [result, ...prev].slice(0, 5);
      return updated;
    });
  };
  
  // Toggle recognition
  const toggleRecognition = async () => {
    if (!isInitialized) return;
    
    if (isRunning) {
      recognitionService.stop();
      setIsRunning(false);
    } else {
      await recognitionService.start(handleRecognitionResult);
      setIsRunning(true);
    }
  };
  
  return (
    <div className="w-full max-w-6xl mx-auto animate-fade-in">
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
        <div className="lg:col-span-3">
          <div className="glass-card overflow-hidden relative">
            {error && (
              <div className="absolute inset-0 flex items-center justify-center bg-background/90 z-10 p-6">
                <div className="flex flex-col items-center text-center">
                  <AlertCircle className="h-10 w-10 text-destructive mb-4" />
                  <h3 className="text-lg font-semibold mb-2">Camera Error</h3>
                  <p className="text-muted-foreground">{error}</p>
                </div>
              </div>
            )}
            
            {!isCameraLoaded && !error && (
              <div className="absolute inset-0 flex items-center justify-center bg-background/90 z-10">
                <div className="flex flex-col items-center">
                  <Camera className="h-8 w-8 text-primary animate-pulse-subtle mb-4" />
                  <p className="text-muted-foreground">Loading camera...</p>
                </div>
              </div>
            )}
            
            <div className="relative aspect-video bg-black rounded-lg overflow-hidden">
              <video 
                ref={videoRef}
                id="webcam"
                className="w-full h-full object-cover"
                autoPlay
                playsInline
                muted
              />
              
              <canvas 
                ref={canvasRef}
                className="absolute top-0 left-0 w-full h-full"
              />
              
              <div className={`absolute bottom-4 right-4 rounded-full p-1 ${
                isRunning 
                  ? 'bg-primary text-white recognition-active' 
                  : 'bg-muted text-muted-foreground'
              }`}>
                <div className="w-12 h-12 rounded-full flex items-center justify-center">
                  {isRunning ? (
                    <Pause className="h-5 w-5" />
                  ) : (
                    <Play className="h-5 w-5" />
                  )}
                </div>
              </div>
            </div>
            
            <div className="p-4 flex justify-between items-center">
              <h3 className="text-lg font-medium">
                {isRunning ? 'Recognition Active' : 'Recognition Paused'}
              </h3>
              
              <button
                id="startBtn"
                onClick={toggleRecognition}
                disabled={!isInitialized}
                className={`btn-primary flex items-center ${
                  !isInitialized ? 'opacity-50 cursor-not-allowed' : ''
                }`}
              >
                {isRunning ? (
                  <>
                    <Pause className="h-4 w-4 mr-2" />
                    Pause
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4 mr-2" />
                    Start
                  </>
                )}
              </button>
            </div>
            
            <div className="px-4 pb-4">
              <div>
                <div className="text-sm text-muted-foreground">Current Sign</div>
                <div id="sign" className="text-2xl font-bold">
                  {latestResult ? latestResult.gesture : "-"}
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <div className="lg:col-span-2">
          <ResultDisplay 
            latestResult={latestResult}
            recentResults={recentResults}
          />
        </div>
      </div>
    </div>
  );
};

export default RecognitionView;
