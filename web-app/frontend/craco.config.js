module.exports = {
  webpack: {
    configure: (webpackConfig) => {
      webpackConfig.resolve.fallback = {
        ...webpackConfig.resolve.fallback,
        http: false,
        https: false,
        http2: false,
        net: false,
        tls: false,
        crypto: false,
        stream: false,
        path: false,
        zlib: false,
        url: false,
        util: false,
        assert: false,
      };
      return webpackConfig;
    },
  },
};
