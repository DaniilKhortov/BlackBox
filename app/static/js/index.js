var inputText = document.getElementById("messageBoxI")
var outputText = document.getElementById("messageBoxO")
var errMessage = document.getElementById("errMessage")
var At = document.getElementById("At")
var Cs = document.getElementById("Cs")
var PL = document.getElementById("PL")

async function decrypt() {
    try {
        errMessage.innerHTML= " ";
        At.classList.remove("type-sign-At-Highlight")
        Cs.classList.remove("type-sign-Cs-Highlight")
        PL.classList.remove("type-sign-PL-Highlight")
        const response = await fetch('/decrypt', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: inputText.value  })  
        });

        if (!response.ok) {
            throw new Error('Помилка при запиті');
        }
        const data = await response.json();

        if (data.result.Type == "atbash"){
            At.classList.add("type-sign-At-Highlight");
            outputText.value = data.result.Result;
            
        }else if (data.result.Type == "caesar"){
            Cs.classList.add("type-sign-Cs-Highlight");
            outputText.value = data.result.Result;

        }else if (data.result.Type == "pl"){
            PL.classList.add("type-sign-PL-Highlight");
            outputText.value = data.result.Result;

        }else if (data.result.Type == "err"){
            errMessage.innerHTML= data.result.Result;
            outputText.value = " ";

        }
            

        return data.result;
    } catch (error) {
        console.error('Помилка:', error);
    }
  }
