function addButton(label, buttonName) {
    var b = document.createElement("BUTTON");
    b.innerHTML = buttonName;
    b.setAttribute('class', 'toggleable_button label_' + label);
    b.onclick = function() {
        $('.toggleable_button').css({
            border: '2px outset',
            'border-radius': '4px'
        });
        $('.toggleable_button.label_' + label).css({
            border: '2px inset',
            'border-radius': '4px'
        });
        $('.toggleable_div').css('display', 'none');
        $('.toggleable_div.label_' + label).css('display', 'inline');
    };
    b.style.border = '2px outset';
    b.style.borderRadius = '4px';
    b.style.margin = '2px';
    return b;
}


function addLanguageToggleButtons() {
    // If there are any toggleable divs
    $toggleableDivs = $(".contents").find(".toggleable_div")
    if ($toggleableDivs.length) {
        // Add buttons to toggle between C++ and Python
        $(".headertitle").append(addButton("cpp", "C++"))
        $(".headertitle").append(addButton("python", "Python"))
    }
    
    $(".toggleable_button").first().click();
    var $clickDefault = $('.toggleable_button.label_python').first();
    if ($clickDefault.length) {
        $clickDefault.click();
    }
    $clickDefault = $('.toggleable_button.label_cpp').first();
    if ($clickDefault.length) {
        $clickDefault.click();
    }
    return;
}
