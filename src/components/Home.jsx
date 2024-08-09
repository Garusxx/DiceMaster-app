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
  const [gatheringClass, setGatheringClass] = useState(null);

  function gatherDataForClass(event) {
    let classNumber = parseInt(event.target.getAttribute("data-class"));
    setGatheringClass(classNumber);
    console.log("Gathering data for class:", classNumber);
    if (!loading && model && isCameraOn) {
      gatherData(classNumber);
    }
  }

  function gatherData(dataClass) {
    let counter = 0;

    const intervalId = setInterval(() => {
      console.log(`Gathering data: ${counter}`);

      let imageFeatures = tf.tidy(() => {
        let videoFrame = tf.browser.fromPixels(videoRef.current);
        let resizedFrame = tf.image.resizeBilinear(
          videoFrame,
          [224, 224],
          true
        );
        let normalizedFrame = resizedFrame.div(255.0);
        return netFeatureModel.predict(normalizedFrame.expandDims());
      });

      setTrainingDataInput((prev) => [...prev, imageFeatures]);
      setTrainingDataOutputs((prev) => [...prev, dataClass]);

      counter++;
      if (counter === 10) {
        clearInterval(intervalId);
        console.log("Zakończono zbieranie danych.");
        console.log("Training data input:", trainingDataInput);
        console.log("Training data outputs:", trainingDataOutputs);
      }
    }, 1000);
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
        <button
          className="data-collector"
          data-class="1"
          onClick={gatherDataForClass}
        >
          Dice One
        </button>
        <button
          className="data-collector"
          data-class="2"
          onClick={gatherDataForClass}
        >
          Dice Two
        </button>
        <div>
          <button>Train</button>
          <button>Reset</button>
        </div>
      </div>
    </div>
  );
};

export default Home;
