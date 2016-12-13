var gulp = require('gulp')
var g_webpack = require('gulp-webpack')
var webpack = require('webpack')
var path = require('path')
var browserSync = require('browser-sync').create()


var webpack_opts = {
  entry: './streaming-tsne.js',
  plugins: []
}

gulp.task('webpack', function () {
  var return_obj = gulp.src('./src')
    .pipe(g_webpack(webpack_opts))
    .pipe(gulp.dest('dist/'))

  if (process.env.NODE_ENV === 'development') {
    return_obj = return_obj.pipe(browserSync.stream())
  }

  return return_obj
})

if (process.env.NODE_ENV === 'development') {
  webpack_opts.devtool = 'eval-source-map'
  webpack_opts.watch = true
} else {
  webpack_opts.devtool = 'source-map'
  webpack_opts.plugins.concat([
    new webpack.optimize.UglifyJsPlugin(),
    new webpack.optimize.DedupePlugin(),
    new webpack.optimize.AggressiveMergingPlugin()
  ])
}

if (process.env.NODE_ENV === 'development') {
  gulp.task('browser-sync', function () {
    browserSync.init({
      server: {
        baseDir: "./"
      }
    })
    gulp.watch("*.html").on('change', browserSync.reload);
  })

  gulp.task('default', ['webpack', 'browser-sync'])
}
