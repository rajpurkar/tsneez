# tsneez
## High Dimensional Visualization Simplified

### Example 1: Data passed as high dimensional pairwise dissimilar vectors
Import tsneez.js into your document: <script src='/tsneez/dist/tsneez.js'></script> and then is some of the usage code.

```javascript
// Hyper parameters
var opt = {}
opt.theta = 0.5 // theta is ...
opt.perplexity = 10 // perplexity is ...
var GRADIENT_STEPS = 5000 // large constant....


var tsneez = new tsneez.TSNEEZ(opt) // create a tsneez instance

// Initialize data. We have four high dimensional pairwise dissimilar vectors
var vects = [
    [0.418, 0.24968, -0.41242, 0.1217, 0.34527],
    [0.013441, 0.23682, -0.16899, 0.40951, 0.63812],
    [0.15164, 0.30177, -0.16763, 0.17684, 0.31719],
    [0.70853, 0.57088, -0.4716, 0.18048, 0.54449],
]
tsneez.initData(vecs)

// A function that applies tsneez algorithm to reduce high dimenionsal vectors
function dimensionReduce(vecs) {

    tsneez.initData(vecs)

    for(var k = 0; k < GRADIENT_STEPS ; k++) {
        tsneez.step() // gradient update
        console.log(`Step : ${k}`)
    }

    var Y = tsneez.Y 

    return Y // Y is an array of 2-dimensional vectors that you can plot.
}

// Execute dimensionality reduction
dimensionReduce(vecs)
```

# Example 2: Data passed to tsneez as a set of high-dimensional Glove vectors
Append the following code within the scripts (<script></script>)tags of your document.

```javascript
// Fetch data from a json file of Glove vectors and execute dimenisonal reductionality
var shortgloveFile = '/tsneez/data/shortglove.json' // A json file of Glove vectors.
document.addEventListener('DOMContentLoaded', function() {
    fetch(shortgloveFile) 
        .then(data => data.json())
        .then((gloves) => {
            var vecs = gloves["vecs"]
            dimensionReduce(vecs)
    })
})
```

===============================================================================================

Fast javascript implementation of T-SNE with tree-based acceleration

[![Standard - JavaScript Style Guide](https://img.shields.io/badge/code%20style-standard-brightgreen.svg)](http://standardjs.com/)