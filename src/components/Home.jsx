import React, { useRef, useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";

const Home = () => {
  const videoRef = useRef(null);
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [trainingDataInput, setTrainingDataInput] = useState([]);
  const [trainingDataOutputs, setTrainingDataOutputs] = useState([]);
  const [model, setModel] = useState(tf.sequential());
  const [netFeatureModel, setNetFeatureModel] = useState(null);
  const [loading, setLoading] = useState(true);
  const [classNames, setClassNames] = useState([]);


  // model.add(
  //   tf.layers.dense({ inputShape: [1024], units: 128, activation: "relu" })
  // );
  // model.add(
  //   tf.layers.dense({ units: classNames.length, activation: "softmax" })
  // );

  // model.summary();

  // model.compile({
  //   optimizer: "adam",
  //   loss: (classNames.length === 2) ? "binaryCrossentropy" : "categoricalCrossentropy",
  //   metrics: ["accuracy"],
  // })

 

  function getherDataForClass(event) {
    let classNumber = parseInt(event.target.getAttribute("data-class"));
    console.log("Ghetering :" + classNumber);
  }

  function toggleCamera() {
    setIsCameraOn((prevState) => !prevState);
  }

  async function loadMobileNetFeatureModel() {
    try {
      const loadedModel = await tf.loadGraphModel(
        "https://www.kaggle.com/models/google/inception-v3/TfJs/classification/2",
        { fromTFHub: true }
      );

      setNetFeatureModel(loadedModel);

      tf.tidy(() => {
        if (loadedModel) {
          let answer = loadedModel.predict(tf.zeros([1, 224, 224, 3]));
          console.log(answer);
        } else {
          console.error("Model is null");
        }
      });
    } catch (error) {
      console.error("Error loading model:", error);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadMobileNetFeatureModel();
  }, []);

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
      {loading && <p>Loading model...</p>}
      {!loading && netFeatureModel && <p>Model załadowany poprawnie</p>}
      {!loading && !netFeatureModel && <p>Nie udało się załadować modelu</p>}
      <div>
        <button onClick={toggleCamera}>Camera On/Off</button>
      </div>
      <video ref={videoRef} autoPlay />
      <div>
        <button data-class="1" onClick={getherDataForClass}>Dice One</button>
        <button data-class="2" onClick={getherDataForClass}>Dice Two</button>
        <div>
          <button>Trein</button>
          <button>Reset</button>
        </div>
      </div>
    </div>
  );
};

export default Home;
