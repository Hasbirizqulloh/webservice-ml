const tf = require('@tensorflow/tfjs-node');
const InputError = require('../exceptions/InputError');

async function predictClassification(model, image) {
  try {
    const tensor = tf.node.decodeJpeg(image).resizeNearestNeighbor([224, 224]).expandDims().toFloat();

    const prediction = model.predict(tensor);
    const score = await prediction.data();
    const confidenceScore = Math.max(...score) * 100;

    let label, suggestion;

    if (confidenceScore > 50) {
      label = 'Cancer';
      suggestion = 'Segera periksa ke dokter untuk mendapatkan pemeriksaan lebih lanjut dan penanganan yang tepat.';
    } else {
      label = 'Non-cancer';
      suggestion = 'Tetap jaga kesehatan kulit Anda dan lakukan pemeriksaan rutin untuk memastikan kondisi kulit tetap sehat.';
    }

    return { label, suggestion };
  } catch (error) {
    throw new InputError(`Terjadi kesalahan dalam melakukan prediksi`);
  }
}

module.exports = predictClassification;
