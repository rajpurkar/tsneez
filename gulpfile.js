var gulp = require('gulp')
var pug = require('gulp-pug')
var g_webpack = require('gulp-webpack')
var webpack = require('webpack')
var path = require('path')
var browserSync = require('browser-sync').create()
var stylus = require('gulp-stylus')
var minify = require('gulp-minify')
var bower = require('gulp-bower')
var ghPages = require('gulp-gh-pages')

var name = 't-sneez'
var build_dir = name + '/'

var webpack_opts = {
  context: path.join(__dirname, 'src/'),
  entry: {},
  output: {
    filename: "[name].js"
  },
  plugins: [
    new webpack.optimize.UglifyJsPlugin({
      include: /\.min\.js$/,
      minimize: true
    })
  ]
}
webpack_opts.entry[name] = './' + name + '.js'
webpack_opts.entry[name + '.min'] = './' + name + '.js'


gulp.task('webpack', function () {
  return gulp.src('.')
    .pipe(g_webpack(webpack_opts))
    .pipe(gulp.dest(build_dir + 'dist/'))
})

gulp.task('webpack_with_watch', function () {
  webpack_opts.watch = true
  
  var return_obj = gulp.src('.')
    .pipe(g_webpack(webpack_opts))
    .pipe(gulp.dest(build_dir + 'dist/'))

  return return_obj.pipe(browserSync.stream())
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

gulp.task('js', function () {
  return gulp.src('./views/js/*')
    .pipe(minify())
    .pipe(gulp.dest('./' + build_dir + 'javascripts/'))
})

gulp.task('html', function () {
  return gulp.src('views/index.pug')
    .pipe(pug({locals: {bd: '/' + build_dir}}))
    .pipe(gulp.dest('./' + build_dir))
})

gulp.task('browser-sync', ['preprocess'], function () {
  browserSync.init({
    server: {
      baseDir: './'
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

gulp.task('deploy', function () {
  return gulp.src('./' + build_dir + '**/*')
    .pipe(ghPages())
})

gulp.task('preprocess', ['css', 'html', 'js', 'bower', 'copy_data'])
gulp.task('default', ['webpack', 'preprocess'])
gulp.task('server', ['browser-sync', 'preprocess', 'css-watch', 'js-watch', 'html-watch', 'webpack_with_watch'])
