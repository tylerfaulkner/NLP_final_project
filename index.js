url = "http://nlp-prototype.tyler-faulkner.com:5000/generateSummary";

function generateSumm(button) {
  var text = button.textContent;
  text = text.replace(/ /g, "-");
  document.getElementById("status").innerHTML = "Generating summary...";

  $.get(url, { filename: text }, function (data) {
    document.getElementById("status").innerHTML = "Done!";
    summary = toText(data);
    document.getElementById("summaryBox").innerHTML = summary;
  });
  console.log(text);
}

function toText(text) {
  var tmp = document.createElement("div");
  tmp.appendChild(document.createTextNode(text));
  return tmp.innerHTML;
}
