/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./templates/**/*.html",
    "./static/src/**/*.js",
    "./node_modules/flowbite/**/*.js"
  ],
  theme: {
    colors: {
      transparent: 'transparent',
      current: 'currentColor',
      'sidebar': '#253150',
      'secondary': '#96B5DA',
      'background': '#EAEEF2'
    },
    extend: {},
  },
  plugins: [
    require("flowbite/plugin")
  ],
}

