var gulp = require('gulp')
var pug = require('gulp-pug')
var g_webpack = require('gulp-webpack')
var webpack = require('webpack')
var path = require('path')
var browserSync = require('browser-sync').create()
var stylus = require('gulp-stylus')
var minify = require('gulp-minify')
var bower = require('gulp-bower')

var build_dir = 'streaming-tsne-js/'

var webpack_opts = {
  context: path.join(__dirname, 'src/'),
  entry: './streaming-tsne.js',
  output: {
    filename: 'streaming-tsne.js'
  },
  plugins: []
}

gulp.task('webpack', function () {
  var return_obj = gulp.src('.')
    .pipe(g_webpack(webpack_opts))
    .pipe(gulp.dest(build_dir + 'dist/'))

  if (process.env.NODE_ENV === 'development') {
    return_obj = return_obj.pipe(browserSync.stream())
  }

  return return_obj
})

gulp.task('copy_data', function () {
  gulp
    .src('data/*')
    .pipe(gulp.dest('./' + build_dir + 'data/'))
})

gulp.task('bower', function () {
  return bower()
    .pipe(gulp.dest('./' + build_dir + 'bower_components/'))
})

gulp.task('css', function () {
  return gulp.src('./views/styles/*.styl')
    .pipe(stylus())
    .pipe(gulp.dest('./' + build_dir + 'stylesheets'))
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

gulp.task('js', function () {
  return gulp.src('./views/js/*')
    .pipe(minify())
    .pipe(gulp.dest('./' + build_dir + 'javascripts/'))
})

gulp.task('html', function () {
  return gulp.src('views/index.pug')
    .pipe(pug())
    .pipe(gulp.dest('./' + build_dir))
})

gulp.task('browser-sync', ['preprocess'], function () {
  browserSync.init({
    server: {
      baseDir: build_dir
    }
  })
  gulp.watch(build_dir + "**/*").on('change', browserSync.reload);
})

gulp.task('html-watch', ['html'], function () {
  gulp.watch('./views/**/*.pug', ['html'])
})

gulp.task('css-watch', ['css'], function () {
  gulp.watch('./views/styles/**/*.styl', ['css'])
})

gulp.task('js-watch', ['js'], function () {
  gulp.watch('./views/js/**/*.js', ['js'])
})

gulp.task('preprocess', ['css', 'html', 'js', 'bower', 'copy_data'])

gulp.task('default', ['webpack', 'browser-sync', 'preprocess', 'css-watch', 'js-watch', 'html-watch'])