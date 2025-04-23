
import Hero from "../components/Hero";
import Navbar from "../components/Navbar";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Link } from "react-router-dom";

const Index = () => {
  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      <Hero />
      
      <div className="container mx-auto px-4 sm:px-6 py-16 flex-grow">
        <div className="text-center max-w-3xl mx-auto mb-10">
          <h2 className="text-3xl sm:text-4xl font-bold mb-6">
            Breaking Down Communication Barriers
          </h2>
          <p className="text-muted-foreground mb-8">
            Our sign language recognition system uses TensorFlow.js to recognize and translate 
            hand gestures representing American Sign Language letters. Try our real-time 
            recognition system with your webcam!
          </p>
          
          <Link to="/recognition" className="btn-primary inline-flex items-center px-6 py-3">
            Try Sign Recognition
          </Link>
        </div>
        
        <div className="grid md:grid-cols-3 gap-6 mt-10">
          <Card>
            <CardHeader>
              <CardTitle>Real-time Recognition</CardTitle>
            </CardHeader>
            <CardContent>
              <CardDescription>
                Our system processes webcam video in real-time to recognize hand gestures 
                representing letters of the American Sign Language alphabet.
              </CardDescription>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader>
              <CardTitle>TensorFlow.js Powered</CardTitle>
            </CardHeader>
            <CardContent>
              <CardDescription>
                We use TensorFlow.js to run machine learning models directly in your browser, 
                ensuring your data never leaves your device.
              </CardDescription>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader>
              <CardTitle>Audio Feedback</CardTitle>
            </CardHeader>
            <CardContent>
              <CardDescription>
                When a sign is recognized, the system speaks the corresponding letter aloud, 
                providing immediate audio feedback similar to your original JavaScript implementation.
              </CardDescription>
            </CardContent>
          </Card>
        </div>
      </div>
      
      {/* Footer */}
      <footer className="bg-secondary py-10 mt-auto">
        <div className="container mx-auto px-4 sm:px-6">
          <div className="text-center text-muted-foreground">
            <p>© {new Date().getFullYear()} GestureSpeak. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Index;
