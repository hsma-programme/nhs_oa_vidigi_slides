window.addEventListener("load", function(event) {
  iFrameResize({
    // log: true, // Use for debugging
    minHeight: 140,
    sizeWidth: true,
    maxWidth: 800,
    maxHeight: 675,
    widthCalculationMethod: 'rightMostElement' // Or 'bodyOffset'
  }, ".stlite");
});
