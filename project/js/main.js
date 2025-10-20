document.addEventListener("DOMContentLoaded", () => {
  const yearSpan = document.getElementById("copyright");
  if (yearSpan) {
    const year = new Date().getFullYear();
    yearSpan.textContent = `© ${year} House Maxify. All rights reserved.`;
  }

  const smoothLinks = document.querySelectorAll('a[href^="#"]');
  smoothLinks.forEach((link) => {
    link.addEventListener("click", (event) => {
      const targetId = link.getAttribute("href").substring(1);
      const target = document.getElementById(targetId);
      if (target) {
        event.preventDefault();
        target.scrollIntoView({ behavior: "smooth" });
      }
    });
  });
});
