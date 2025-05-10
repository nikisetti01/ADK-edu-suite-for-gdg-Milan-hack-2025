document.addEventListener('DOMContentLoaded', () => {
    const skipButton = document.querySelector('.skip-button');
    const finishButton = document.querySelector('.finish-exam-button');
    const progressBar = document.querySelector('.progress .progress');
    let currentQuestion = 1;
    const totalQuestions = 12;

    function updateProgressBar() {
        const progressPercentage = (currentQuestion / totalQuestions) * 100;
        progressBar.style.width = `${progressPercentage}%`;
    }

    skipButton.addEventListener('click', () => {
        if (currentQuestion < totalQuestions) {
            currentQuestion++;
            document.querySelector('.question-number').textContent = `Domanda ${currentQuestion}/${totalQuestions}`;
            updateProgressBar();
            // Qui andrebbe la logica per caricare la prossima domanda
            console.log('Salta alla domanda:', currentQuestion);
        } else {
            console.log('Ultima domanda raggiunta.');
        }
    });

    finishButton.addEventListener('click', () => {
        console.log('Esame terminato.');
        // Qui andrebbe la logica per inviare l'esame
        alert('Esame terminato!');
    });

    // Funzionalità dei bottoni azione (solo console log per ora)
    document.querySelector('.chat-button').addEventListener('click', () => {
        console.log('Apri chat.');
    });

    document.querySelector('.record-button').addEventListener('click', () => {
        console.log('Inizia registrazione vocale.');
    });

    document.querySelector('.keyboard-button').addEventListener('click', () => {
        console.log('Apri tastiera.');
    });
});