   const title=document.getElementById("title"); 
const text="Job Prediction Model";
let i=0;
const typeBot=setInterval(() => {
    if(i<text.length){
        title.textContent+=text.charAt(i);
        i++;
    }else{
        clearInterval(typeBot)
    }
    
}, 90);