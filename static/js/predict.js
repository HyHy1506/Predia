// let form =document.querySelector("#formEnterInfoID")
// console.log(form.classList)

// form.addEventListener('submit',e=>{
//     if(!form.checkValidity()){
//     e.preventDefault()
//     }
// form.classList.add("was-validated")
// })
function toggleInputs() {
    var useFileCheckbox = document.getElementById('checkUseFile');
    var inputs = document.querySelectorAll('#so1, #so2, #so3, #so4, #so5');
    var fileInput = document.getElementById('so6');
    
    if (useFileCheckbox.checked) {
        inputs.forEach(function(input) {
            input.disabled = true;
        });
        fileInput.disabled = false;
    } else {
        inputs.forEach(function(input) {
            input.disabled = false;
        });
        fileInput.disabled = true;
    }
}

document.addEventListener('DOMContentLoaded', function() {
    var useFileCheckbox = document.getElementById('checkUseFile');
    useFileCheckbox.checked = false; // Đảm bảo rằng checkbox bị tắt khi trang được tải
    toggleInputs(); // Gọi hàm này để đảm bảo trạng thái ban đầu là chính xác
    useFileCheckbox.addEventListener('change', toggleInputs);
});