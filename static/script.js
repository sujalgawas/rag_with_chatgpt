async function sendMessage() {
    const userInput = document.getElementById("user-input").value;
    if (!userInput) return;

    appendMessage("You", userInput);

    const response = await fetch('http://localhost:5000/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: userInput }),
    });

    const data = await response.json();

    appendMessage("Bot", data.response);

    document.getElementById("user-input").value = "";
}

function appendMessage(sender, message) {
    const chatBox = document.getElementById("chat-box");
    const messageElement = document.createElement("div");
    messageElement.classList.add(sender.toLowerCase());
    messageElement.textContent = `${sender}: ${message}`;
    chatBox.appendChild(messageElement);

    chatBox.scrollTop = chatBox.scrollHeight;
}
