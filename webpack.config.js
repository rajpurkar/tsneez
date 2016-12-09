const path = require('path');

module.exports = {
  entry: './streaming-tsne.js',
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: 'streaming-tsne.js'
  }
};
