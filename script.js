document.addEventListener('DOMContentLoaded', function () {
    const menuLinks = document.querySelectorAll('.menu-link');
    const homeLink = document.getElementById('home-link');
    const teachingLink = document.querySelector('.teaching-link');
    const yearList = document.querySelector('.year-list');

    // Function to show the selected section
    function showSection(sectionId) {
        // Hide all sections
        document.querySelectorAll('.content > section').forEach(section => {
            section.classList.add('hidden');
        });

        // Show the selected section
        document.getElementById(sectionId).classList.remove('hidden');
    }

    // Event listener for menu links
    menuLinks.forEach(link => {
        link.addEventListener('click', function (event) {
            event.preventDefault(); // Prevent the default link behavior
            showSection(this.getAttribute('data-section'));
        });
    });

    // Event listener for the profile picture (home link)
    homeLink.addEventListener('click', function (event) {
        event.preventDefault(); // Prevent the default link behavior
        showSection('home'); // Show the homepage
    });

    // Event listener for the Teaching link
    teachingLink.addEventListener('click', function (event) {
        event.preventDefault(); // Prevent default link behavior
        yearList.classList.toggle('hidden'); // Toggle the visibility of the year list
    });

    // Show the homepage by default
    showSection('home');
});
