
import Navbar from "../components/Navbar";
import RecognitionView from "../components/RecognitionView";

const Recognition = () => {
  return (
    <div className="min-h-screen">
      <Navbar />
      
      <div className="page-container pt-20">
        <div className="mb-6 text-center">
          <h1 className="section-title">Sign Language Recognition</h1>
          <p className="section-subtitle mx-auto">
            Our sign language recognition system detects hand gestures in real-time,
            converts them to letters and speaks them out loud.
          </p>
        </div>
        
        <RecognitionView />
        
        {/* <div className="mt-12 max-w-3xl mx-auto text-center">
          <h3 className="text-xl font-semibold mb-4">How It Works</h3>
          <p className="text-muted-foreground">
            Our system uses MediaPipe for hand tracking and a TensorFlow.js model to recognize 
            hand gestures representing letters. The webcam captures your hand movements, 
            our project recognizes the gestures, and instantly displays the corresponding letter and speaks it out loud.
          </p>
          
          <div className="mt-4 p-4 bg-yellow-100 text-yellow-800 rounded-md text-sm">
            <p><strong>Note:</strong> For the model to work correctly, place your trained model files in the 
            <code className="bg-yellow-200 px-1 rounded">/model</code> directory in your public folder.</p>
            <p className="mt-2">The model should include <code className="bg-yellow-200 px-1 rounded">model.json</code> and 
            corresponding weight files that match the labels ["A", "B", "C", "D", "E", "F"].</p>
          </div>
        </div> */}
      </div>
      
      {/* Footer */}
      <footer className="bg-secondary py-10 mt-20">
        <div className="container mx-auto px-4 sm:px-6">
          <div className="text-center text-muted-foreground">
            <p>© {new Date().getFullYear()} GestureSpeak. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Recognition;
