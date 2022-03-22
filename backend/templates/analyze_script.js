$(document).ready(function () {
  $("#analyzeForm").submit(function (event) {
    var formData = {
      image_path: $("#image_path").val(),
    };

    $.ajax({
      type: "POST",
      url: "/analyze",
      data: formData,
      dataType: "json",
      encode: true,
    }).done(function (data) {
      $('#result').html(`<p class="value3 mt-sm">Prognosis ${data} likely to be SCC</p>`);
    });

    event.preventDefault();
  });


  // Get the modal
var modal = document.getElementById("myModal");

// Get the image and insert it inside the modal - use its "alt" text as a caption
var img = document.getElementById("myImg");
var modalImg = document.getElementById("img01");
var captionText = document.getElementById("caption");
img.onclick = function(){
  modal.style.display = "block";
  modalImg.src = this.src;
  captionText.innerHTML = this.alt;
}

// Get the <span> element that closes the modal
var span = document.getElementsByClassName("close")[0];

// When the user clicks on <span> (x), close the modal
span.onclick = function() {
  modal.style.display = "none";
}

});

