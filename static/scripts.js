/*
* Initializes the theme toggle functionality.
* Checks for a saved theme in localStorage and applies it.
* Adds a click listener to the theme toggle button.
*/
function initThemeToggle() {
    // Find the toggle button in the DOM
    const toggleButton = document.getElementById('theme-toggle');
    
    // If the button doesn't exist on this page, do nothing.
    if (!toggleButton) {
        return;
    }

    // Check if a theme is already saved in the browser's local storage
    const currentTheme = localStorage.getItem('theme');

    // Apply the saved theme on load
    // By default, the site is dark. We ONLY add the class if the theme is 'light'.
    if (currentTheme === 'light') {
        document.body.classList.add('light-mode');
    }

    // Button click listener
    toggleButton.addEventListener('click', function() {
        // Toggle the .light-mode class on the <body> element
        document.body.classList.toggle('light-mode');
        
        let theme = 'dark'; // Default to dark
        // Check if the body now has the .light-mode class
        if (document.body.classList.contains('light-mode')) {
            theme = 'light';
        }
        
        // Save the user's preference to local storage
        localStorage.setItem('theme', theme);
    });
}

/*
* Starts a countdown timer that redirects to the homepage.
*/
function startCountdown(seconds) {
    const countdownElement = document.getElementById('countdown');

    // If the countdown element doesn't exist on this page, do nothing.
    if (!countdownElement)
        return;

    // Set the initial text
    countdownElement.textContent = seconds;
            
    // Start an interval that runs every 1000ms (1 second)
    const interval = setInterval(function() {
        seconds--; // Decrement seconds
        countdownElement.textContent = seconds; // Update the number on screen
        
        // When countdown reaches 0
        if (seconds <= 0) {
            clearInterval(interval); // Stop the countdown
            window.location.href = "/"; // Redirect to the homepage
        }
    }, 1000);
}

// We only call the functions themselves once the DOM has loaded.
// This ensures that the 'theme-toggle' and 'countdown' buttons already exist.
document.addEventListener('DOMContentLoaded', function() {
    initThemeToggle();
    
    // We only call startCountdown if the countdown element exists (which is only on 404.html)
    if (document.getElementById('countdown')) {
        startCountdown(5);
    }
});