document.addEventListener("DOMContentLoaded", function () {
    var script = document.createElement("script");
    script.defer = true;
    script.type = "module";
    script.id = "rankify-widget-script";
    script.src =
      "https://widget.runllm.com";
    script.setAttribute("rankify-name", "Rankify");
    script.setAttribute("rankify-preset", "mkdocs");
    script.setAttribute("rankify-server-address", "https://api.rankify.com");
    script.setAttribute("rankify-assistant-id", "132");
    script.setAttribute("rankify-position", "BOTTOM_RIGHT");
    script.setAttribute("rankify-keyboard-shortcut", "Mod+j");
    script.setAttribute(
      "rankify-slack-community-url",
      ""
    );
  
    document.head.appendChild(script);
  });