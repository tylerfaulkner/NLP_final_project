url = "http://127.0.0.1:5000/generateSummary";

function generateSumm(button) {
  var text = button.textContent;
  text = text.replace(/ /g, "-");

  $.get(url, { filename: text }, function (data) {
    document.getElementById("summaryBox").innerHTML = data;
  });
  console.log(text);
}
