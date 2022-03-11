const allSideMenu = document.querySelectorAll('#sidebar .side-menu.top li a');

allSideMenu.forEach(item=> {
    const li = item.parentElement;

    item.addEventListener('click', function () {
        allSideMenu.forEach(i=> {
            i.parentElement.classList.remove('active');
        })
        li.classList.add('active');
    })
});


// TOGGLE SIDEBAR
const menuBar = document.querySelector('#content nav .bx.bx-menu');
const sideBar = document.getElementById('sidebar')

menuBar.addEventListener('click', function () {
    sideBar.classList.toggle('hide');
})

const searchButton = document.querySelector('#content nav form .form-input button');
const searchButtonIcon = document.querySelector('#content nav form .form-input button .bx');
const searchForm = document.querySelector('#content nav form');

searchButton.addEventListener('click', function (e) {
    if(window.innerWidth < 576) {
        e.preventDefault();
        searchForm.classList.toggle('show');
        if(searchForm.classList.contains('show')) {
            searchButtonIcon.classList.replace('bx-search', 'bx-x');
        } else {
            searchButtonIcon.classList.replace('bx-x', 'bx-search');
        }
    }
})

if(window.innerWidth < 768) {
    sideBar.classList.add('hide');
} else if(window.innerWidth > 576) {
    searchButtonIcon.classList.replace('bx-x', 'bx-search');
    searchForm.classList.remove('show');
}

window.addEventListener('resize', function () {
    if(this.innerWidth > 576) {
        searchButtonIcon.classList.replace('bx-x', 'bx-search');
        searchForm.classList.remove('show');
    }
})


//NAVBAR STICKY
window.onscroll = function() {myFunction()};

var navbar = document.getElementById("navbar");

var sticky = navbar.offsetTop;

function myFunction() {
  if (window.pageYOffset >= sticky) {
    navbar.classList.add("sticky")
  } else {
    navbar.classList.remove("sticky");
  }
}


// IMAGE UPLOAD
const wrapper = document.querySelector(".wrapper");
const fileName = document.querySelector(".file-name");
const defaultBtn = document.querySelector("#default-btn");
const customBtn = document.querySelector("#custom-btn");
const cancelBtn = document.querySelector("#cancel-btn i");
const img = document.getElementById("xray");
var obj = document.getElementById("imgparent");
let regExp = /[0-9a-zA-Z\^\&\'\@\{\}\[\]\,\$\=\!\-\#\(\)\.\%\+\~\_ ]+$/;
function defaultBtnActive(){
  defaultBtn.click();
}
defaultBtn.addEventListener("change", function(){
  let file = this.files[0];

  // Only process image files.
  var extension = file.name.substring(file.name.lastIndexOf('.'));
  var validFileType = ".jpg , .jpeg, .png, .dcm";
  if (validFileType.toLowerCase().indexOf(extension) < 0) {
      alert("Please select a valid file type. The supported file types are .jpg , .png , .dcm");
      return false;
  }

  if(file){
    obj.appendChild(img)
    const reader = new FileReader();
    reader.onload = function(){
      var result = reader.result;
      img.src = result;
      wrapper.classList.add("active");
    }
    cancelBtn.addEventListener("click", function(){
      obj.removeChild(img)
      wrapper.classList.remove("active");
    })
    reader.readAsDataURL(file);
  } else {
    img.src = "";
  }
  if(this.value){
    let valueStore = this.value.match(regExp);
    fileName.textContent = valueStore;
  }
});


// PROGESS BAR
function additionalElems(progressElement, value, diseaseName) {
  var valueChild = document.createElement('span');
  valueChild.className = 'progress-bar__value';
  valueChild.innerHTML = value +'%';
  progressElement.appendChild(valueChild);
  
  var barChild = document.createElement('div');
  barChild.className = 'progress-bar__bar';
  progressElement.appendChild(barChild);
  
  var barInnerChild = document.createElement('div');
  barInnerChild.className = 'progress-bar__bar-inner';
  barInnerChild.style.width = value + '%';
  barChild.appendChild(barInnerChild);
 
  var diseaseChild = document.createElement('span');
  diseaseChild.className = 'progress-bar__disease';
  diseaseChild.innerHTML = diseaseName;
  progressElement.appendChild(diseaseChild);
}

var progressBar = document.querySelectorAll('.progress-bar');

progressBar.forEach(function(item) {
  
  var self = item,
      barValue = self.getAttribute('data-value'),
      diseaseValue = self.getAttribute('data-disease'),
      effectNum = self.getAttribute('data-effect');
  
  additionalElems(self, barValue, diseaseValue);
  
  self.className += ' progress-bar--' + effectNum;
  
  var valueElem = self.querySelector('.progress-bar__value');
    
  valueElem.className += ' progress-bar__value--' + effectNum;
  
  var barElem = self.querySelector('.progress-bar__bar');
  
  barElem.className += ' progress-bar__bar--' + effectNum;
  
  var barInnerElem = self.querySelector('.progress-bar__bar-inner');
  
  barInnerElem.className += ' progress-bar__bar-inner--' + effectNum;
  
  var diseaseElem = self.querySelector('.progress-bar__disease');
    
  diseaseElem.className += ' progress-bar__disease--' + effectNum;
  
  if(self.classList.contains('progress-bar--aligned-values')) {
     valueElem.style.left = barValue + '%';
     valueElem.classList.add('progress-bar__value--aligned-value');
  }
  
  if(self.classList.contains('progress-bar--no-overflow')) {
     barElem.classList.add('progress-bar__bar--no-overflow');
  }
  
})

function animationToggle(progressElement, delay) {
  
    var diseaseElem = progressElement.querySelector('.progress-bar__disease'),
    valueElem = progressElement.querySelector('.progress-bar__value'),
    diseaseBar = progressElement.querySelector('.progress-bar__bar-inner');

    diseaseElem.classList.remove('js-animated');
    diseaseElem.classList.remove('js-animated');
    diseaseBar.classList.remove('js-animated');
  
  setTimeout(function() {
    diseaseElem.classList.add('js-animated');
    valueElem.classList.add('js-animated');
    diseaseBar.classList.add('js-animated');
  }, delay);
}

function onloadAnimation() {
  
  progressBar.forEach(function(item) {
    animationToggle(item, 500)
  });
  
}

document.addEventListener("DOMContentLoaded", onloadAnimation());


