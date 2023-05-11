url = "http://nlp-prototype.tyler-faulkner.com:5000/generateSummary";

function generateSumm(button) {
  var text = button.textContent;
  text = text.replace(/ /g, "-");
  document.getElementById("status").innerHTML = "Generating summary...";

  $.get(url, { filename: text }, function (data) {
    document.getElementById("summaryBox").innerHTML = text(data).html();
    document.getElementById("status").innerHTML = "Done!";
  });
  console.log(text);
}
