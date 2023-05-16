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
  return tmp.innerHTML.replace(/\\uffd/g, "");
}

function generateSummFromFile() {
  document.getElementById("status").innerHTML = "Generating summary...";
  var file = document.getElementById("fileInput").files[0];
  var reader = new FileReader();
  reader.readAsText(file);
  reader.onload = function () {
    fetch(url + "FromFile", {
      method: "POST",
      body: reader.result,
    })
      .then((response) => response.text())
      .then((data) => {
        document.getElementById("status").innerHTML = "Done!";
        document.getElementById("summaryBox").innerHTML = toText(data);
      }
      );
  }


}