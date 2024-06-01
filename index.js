
import {MnistData} from "/data.js";
var canvas, ctx, saveButton, clearButton;
var pos = {x:0, y:0};
var rawImage;
var model;



function getModel(){
  model = tf.sequential();

  model.add( tf.layers.conv2d({ inputShape:[28,28,1] , kernelSize:3 , filters:16 , activation:"relu"}));
  model.add( tf.layers.maxPooling2d({ poolSize:[2,2]}));

  model.add( tf.layers.conv2d({ kernelSize:3 , filters:32 , activation:"relu"}));
  model.add( tf.layers.maxPooling2d({ poolSize:[2,2]}));

  model.add( tf.layers.conv2d({ kernelSize:3 , filters:64 , activation:"relu"}));
  model.add( tf.layers.maxPooling2d({ poolSize:[2,2]}));

  model.add( tf.layers.flatten() );

  model.add( tf.layers.dense({ units:32 , activation:"relu"}));
  
  model.add(tf.layers.dense({ units:10 , activation:"softmax"}));

  model.compile({optimizer:tf.train.adam() , loss:"categoricalCrossentropy" , metrics:['accuracy']});

  return model;

}

async function trainModel(model,data){
  // CALLBACKS
  const metrics = ['accuracy','val_accuracy','loss','val_loss'];
  const container = { name:"Model" , styles:{height:"700px"}};
  const fitCallbacks = tfvis.show.fitCallbacks(container,metrics);

  // TRAIN 
  const trainDataSize = 5000;
  const testDataSize = 1500;
  const batch_Size = 512;

  const [trainX,trainY] = tf.tidy( ()=>{
    const d = data.nextTrainBatch(trainDataSize);

    return[ d.xs.reshape([trainDataSize,28,28,1]),d.labels];
  });

  const [testX,testY] = tf.tidy( ()=>{
    const d = data.nextTestBatch(testDataSize);

    return[ d.xs.reshape([testDataSize,28,28,1]),d.labels];
  });

  return model.fit( trainX, trainY, {
    validationData:[testX,testY],
    epochs:20,
    batchSize: batch_Size,
    shuffle: true,
    callbacks: fitCallbacks
  });


}

// THE CANVAS 


function setPosition(e){
	pos.x = e.clientX-100;
	pos.y = e.clientY-100;
}
    
function draw(e) {
	if(e.buttons!=1) return;
	ctx.beginPath();
	ctx.lineWidth = 24;
	ctx.lineCap = 'round';
	ctx.strokeStyle = 'white';
	ctx.moveTo(pos.x, pos.y);
	setPosition(e);
	ctx.lineTo(pos.x, pos.y);
	ctx.stroke();
	rawImage.src = canvas.toDataURL('image/png');
}
    

function erase() {
	ctx.fillStyle = "black";
	ctx.fillRect(0,0,280,280);
}
    
function save() {
	var raw = tf.browser.fromPixels(rawImage,1);
	var resized = tf.image.resizeBilinear(raw, [28,28]);
	var tensor = resized.expandDims(0);
    var prediction = model.predict(tensor);
    var pIndex = tf.argMax(prediction, 1).dataSync();
    
	alert(pIndex);
}

function init() {
	canvas = document.getElementById('canvas');
	rawImage = document.getElementById('canvasimg');
	ctx = canvas.getContext("2d");
	ctx.fillStyle = "black";
	ctx.fillRect(0,0,280,280);
	canvas.addEventListener("mousemove", draw);
	canvas.addEventListener("mousedown", setPosition);
	canvas.addEventListener("mouseenter", setPosition);
	saveButton = document.getElementById('sb');
	saveButton.addEventListener("click", save);
	clearButton = document.getElementById('cb');
	clearButton.addEventListener("click", erase);
}

async function run(){
  const data = new MnistData();
  await data.load();
  const model = getModel();
  await trainModel(model,data);

  init();
}

document.addEventListener('DOMContentLoaded', run);
