(function () {
  'use strict';

  // Upload dropzone: show file name, image preview, drag highlight
  var drop = document.querySelector('.drop');
  if (drop) {
    var input = drop.querySelector('input[type="file"]');
    var idle = drop.querySelector('[data-drop-idle]');
    var nameEl = drop.querySelector('[data-drop-name]');
    var thumb = drop.querySelector('[data-drop-thumb]');

    var showFile = function () {
      var file = input.files && input.files[0];
      if (!file) return;
      nameEl.textContent = file.name;
      nameEl.hidden = false;
      idle.hidden = true;
      if (thumb && file.type.indexOf('image/') === 0) {
        thumb.src = URL.createObjectURL(file);
        thumb.hidden = false;
      }
      var err = document.getElementById('form-error');
      if (err) err.classList.remove('show');
    };

    input.addEventListener('change', showFile);
    ['dragenter', 'dragover'].forEach(function (t) {
      drop.addEventListener(t, function (e) { e.preventDefault(); drop.classList.add('drag'); });
    });
    ['dragleave', 'drop'].forEach(function (t) {
      drop.addEventListener(t, function () { drop.classList.remove('drag'); });
    });
    drop.addEventListener('drop', function (e) {
      e.preventDefault();
      if (e.dataTransfer && e.dataTransfer.files.length) {
        input.files = e.dataTransfer.files;
        showFile();
      }
    });
  }

  // Busy state on submit buttons that pass validation
  document.querySelectorAll('form').forEach(function (form) {
    form.addEventListener('submit', function (e) {
      if (e.defaultPrevented) return;
      var btn = form.querySelector('button[data-busy]');
      if (btn) {
        // Delay so the browser still submits the form before the button disables
        setTimeout(function () { btn.disabled = true; btn.textContent = btn.getAttribute('data-busy') + '...'; }, 0);
      }
    });
  });

  // Home diagram: chips and zones highlight each other
  var stage = document.getElementById('stage');
  if (stage) {
    var setZone = function (zone, on) {
      stage.querySelectorAll('[data-zone="' + zone + '"]').forEach(function (el) {
        el.classList.toggle('on', on);
      });
    };
    stage.querySelectorAll('.chip, .zone').forEach(function (el) {
      var zone = el.getAttribute('data-zone');
      el.addEventListener('mouseenter', function () { setZone(zone, true); });
      el.addEventListener('mouseleave', function () { setZone(zone, false); });
      el.addEventListener('focus', function () { setZone(zone, true); });
      el.addEventListener('blur', function () { setZone(zone, false); });
    });
  }
})();
