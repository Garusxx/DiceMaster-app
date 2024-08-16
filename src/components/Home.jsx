import React, { useRef, useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";

const Home = () => {
  const videoRef = useRef(null);
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [trainingDataInput, setTrainingDataInput] = useState([]);
  const [trainingDataOutputs, setTrainingDataOutputs] = useState([]);
  const [netFeatureModel, setNetFeatureModel] = useState(null);
  const [loading, setLoading] = useState(true);
  const [isGatheringData, setIsGatheringData] = useState(false);
  const [model, setModel] = useState(tf.sequential());
  const [predict, setPredict] = useState(true);

  model.add(
    tf.layers.dense({ units: 128, inputShape: [1001], activation: "relu" })
  );
  model.add(tf.layers.dense({ units: 2, activation: "softmax" }));

  model.compile({
    optimizer: tf.train.adam(),
    loss: "binaryCrossentropy",
    metrics: ["accuracy"],
  });

  function logProggress(epoch, logs) {
    console.log(
      "Epoch: " + epoch + " Loss: " + logs.loss + " Accuracy: " + logs.acc
    );
  }

  function reset() {
    setTrainingDataInput([]);
    setTrainingDataOutputs([]);
  }


  function predictLoop() {
    if (predict) {
      tf.tidy(() => {
        let videoFrameAsTensor = tf.browser.fromPixels(
          videoRef.current.div(255)
        );
        let resizedFrame = tf.image.resizeBilinear(
          videoFrameAsTensor,
          [224, 224],
          true
        );

        let imageFeatures = netFeatureModel.predict(resizedFrame.expandDims());
        let prediction = model.predict(imageFeatures);
        let predictingArray = prediction.dataSync();

        console.log(
          "Prediction:",
          +predictingArray[0].toFixed(2) + " " + predictingArray[1].toFixed(2)
        );
      });
    }

    window.requestAnimationFrame(predictLoop);
  }

  async function trainModel() {
    setPredict(false);
    tf.util.shuffle(trainingDataInput, trainingDataOutputs);
    let outputAsTensor = tf.tensor1d(trainingDataOutputs, "int32");
    console.log("Output as tensor:", outputAsTensor);
    let oenHotOutput = tf.oneHot(outputAsTensor, 2);
    console.log("One hot output:", oenHotOutput);
    let inputTensor = tf.stack(trainingDataInput);
    console.log("Input tensor:", inputTensor);
    let results = await model.fit(inputTensor, oenHotOutput, {
      shuffle: true,
      epochs: 10,
      batchSize: 5,
      callbacks: { onEpochEnd: logProggress },
    });

    outputAsTensor.dispose();
    oenHotOutput.dispose();
    inputTensor.dispose();
    setPredict(true);
    predictLoop();
  }

  function gatherDataForClass(event) {
    const dataClass = parseInt(event.target.getAttribute("data-class"));
    console.log("Set class for gathering data:", dataClass);

    if (!loading && netFeatureModel && isCameraOn) {
      gatherData(dataClass);
    }
  }

  function gatherData(dataClass) {
    let counter = 0;
    setIsGatheringData(true);
    const intervalId = setInterval(() => {
      console.log(`Gathering data for class ${dataClass}: ${counter}`);

      tf.tidy(() => {
        const videoFrame = tf.browser.fromPixels(videoRef.current);
        const resizedFrame = tf.image.resizeBilinear(
          videoFrame,
          [224, 224],
          true
        );
        const normalizedFrame = resizedFrame.div(255.0);
        const imageFeatures = netFeatureModel.predict(
          normalizedFrame.expandDims()
        );

        setTrainingDataInput((prev) => [...prev, imageFeatures]);
        setTrainingDataOutputs((prev) => [...prev, dataClass]);
      });

      counter++;
      if (counter === 10) {
        clearInterval(intervalId);
        console.log("Finished gathering data.");
        setIsGatheringData(false);
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
    console.log("Training data input:", trainingDataInput);
    console.log("Training data outputs:", trainingDataOutputs);
  }, [trainingDataInput, trainingDataOutputs]);

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
          disabled={isGatheringData || loading}
        >
          Dice One
        </button>
        <button
          className="data-collector"
          data-class="2"
          onClick={gatherDataForClass}
          disabled={isGatheringData || loading}
        >
          Dice Two
        </button>
        <div>
          <button onClick={trainModel}>Train</button>
          <button onClick={reset}>Reset</button>
        </div>
      </div>
    </div>
  );
};

export default Home;
