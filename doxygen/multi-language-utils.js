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

function buttonsToAdd($elements, $heading, $type) {
    if ($elements.length === 0) {
        $elements = $("" + $type + ":contains(" + $heading.html() + ")").parent().prev("div.newInnerHTML");
    }
    var arr = jQuery.makeArray($elements);
    var seen = {};
    arr.forEach(function(e) {
        var txt = e.innerHTML;
        if (!seen[txt]) {
            $button = addButton(e.title, txt);
            if (Object.keys(seen).length == 0) {
                var linebreak1 = document.createElement("br");
                var linebreak2 = document.createElement("br");
                ($heading).append(linebreak1);
                ($heading).append(linebreak2);
            }
            ($heading).append($button);
            seen[txt] = true;
        }
    });
    return;
}

function addLanguageToggleButtons() {
    // Search for H1 headings (in GeNN documentation these are the only heading really used)
    $smallerHeadings = $(".contents").first().find("h1");
    if ($smallerHeadings.length) {
        $smallerHeadings.each(function() {
            var $elements = $(this).nextUntil("h1").filter("div.newInnerHTML");
            buttonsToAdd($elements, $(this), "h1");
        });
    } else {
        var $elements = $(".contents").first().find("div.newInnerHTML");
        buttonsToAdd($elements, $heading, "h2");
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
