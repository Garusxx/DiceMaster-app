import React, { useRef, useEffect, useState } from "react";

const Home = () => {
  const videoRef = useRef(null);
  const [isCameraOn, setIsCameraOn] = useState(false);

  function toggleCamera() {
    setIsCameraOn((prevState) => !prevState);
  }

  useEffect(() => {
    const constraints = { video: true };

    if (isCameraOn) {
      navigator.mediaDevices
        .getUserMedia(constraints)
        .then((stream) => {
          videoRef.current.srcObject = stream;
        })
        .catch((error) => {
          console.error("Error accessing webcam:", error);
        });
    } else {
      if (videoRef.current && videoRef.current.srcObject) {
        let tracks = videoRef.current.srcObject.getTracks();
        tracks.forEach((track) => track.stop());
        videoRef.current.srcObject = null;
      }
    }
  }, [isCameraOn]);

  return (
    <div>
      <h1>DiceMaster 2000</h1>
      <div>
        <button onClick={toggleCamera}>Camera On/Off</button>
      </div>
      <video ref={videoRef} autoPlay />
      <div>
        <button>Dice One</button>
        <button>Dice Two</button>
        <div>
          <button>Trein</button>
          <button>Reset</button>
        </div>
      </div>
    </div>
  );
};

export default Home;
