let form =document.querySelector("#formEnterInfoID")
console.log(form.classList)

form.addEventListener('submit',e=>{
    if(!form.checkValidity()){
    e.preventDefault()
    }
form.classList.add("was-validated")
})