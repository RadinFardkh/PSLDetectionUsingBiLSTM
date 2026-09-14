module.exports = function (api) {
  api.cache(true)
  return {
    presets: ['babel-preset-expo'],
    plugins: [['react-native-worklets-core/plugin', { globals: ['__scanFaces', '__runOnFrame'] }]],
  }
}
