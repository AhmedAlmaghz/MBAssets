/* MaghzBook URL resolver — one rule for every internal link.
 *
 *   resolve(path) = APP_URL ? APP_URL + '/' + root + path : root + path
 *
 * APP_URL sources, in priority order:
 *   1. window.APP_URL            (set by the app shell before page load)
 *   2. process.env.APP_URL       (Node / build tooling)
 *   3. <meta name="app-url">    (injected at build time by tools/resolve_urls.py)
 * `root` comes from <meta name="mb-root"> — used only by the project-root copy
 * of the marketing page, which lives one level above the bundle.
 *
 * Works as a browser script, a CommonJS module and a global object.
 * No dependencies. All landing pages carry an inline copy of this logic so
 * they stay single-file.
 */
(function (global) {
  'use strict';

  var ATTR = { A: 'href', IMG: 'src', SOURCE: 'src', LINK: 'href',
               META: 'content', SCRIPT: 'src', VIDEO: 'src', AUDIO: 'src' };

  function meta(name) {
    if (typeof document === 'undefined') return '';
    var el = document.querySelector('meta[name="' + name + '"]');
    return (el && el.getAttribute('content')) || '';
  }

  function envValue() {
    var fromGlobal = global && global.APP_URL;
    var fromNode = (typeof process !== 'undefined' && process.env && process.env.APP_URL);
    var fromMeta = meta('app-url');
    return String(fromGlobal || fromNode || fromMeta || '').trim().replace(/\/+$/, '');
  }

  function rootValue() {
    return String(meta('mb-root') || '').replace(/^\/+/, '').replace(/\/*$/, '/');
  }

  function build() {
    var base = envValue();
    var r = rootValue();
    // rootValue() returns '/' for an empty meta -> normalise to ''
    if (r === '/') r = '';
    return { base: base, root: r };
  }

  function resolve(path, cfg) {
    if (!path) return path;
    if (/^(?:[a-z][a-z0-9+.-]*:|\/\/|#|data:|mailto:|tel:)/i.test(path)) return path;
    var c = cfg || build();
    var p = c.root + String(path).replace(/^\/+/, '');
    return c.base ? c.base + '/' + p : p;
  }

  function apply(node, cfg) {
    if (typeof document === 'undefined') return 0;
    var c = cfg || build();
    var list = (node || document).querySelectorAll('[data-mb-path]');
    Array.prototype.forEach.call(list, function (el) {
      var attr = el.getAttribute('data-mb-attr') || ATTR[el.tagName] || 'href';
      el.setAttribute(attr, resolve(el.getAttribute('data-mb-path'), c));
    });
    return list.length;
  }

  var api = {
    ENV: 'APP_URL',
    getAppUrl: function () { return envValue(); },
    getRoot: function () { return build().root; },
    build: build,
    resolve: resolve,
    apply: apply
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (global) global.MaghzBookURL = api;

  if (typeof document !== 'undefined' && api.getAppUrl()) {
    if (document.readyState === 'loading')
      document.addEventListener('DOMContentLoaded', function () { apply(); });
    else apply();
  }
})(typeof window !== 'undefined' ? window : (typeof globalThis !== 'undefined' ? globalThis : this));
