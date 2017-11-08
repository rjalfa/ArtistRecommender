String.prototype.getInitials = function(glue){
    if (typeof glue == "undefined") {
        var glue = true;
    }

    var initials = this.replace(/[^a-zA-Z- ]/g, "").match(/\b\w/g);
    
    if (glue) {
        return initials.join('');
    }

    return  initials;
};

String.prototype.capitalize = function(){
    return this.toLowerCase().replace( /\b\w/g, function (m) {
        return m.toUpperCase();
    });
};

var mbids = [];
var selected = [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false];
buildDefaultCells = function() {
	$('div.song-div').each(function() {
		eid = $(this).attr('id');
		$(this).html($(this).html() + "<i class='serial'>" + eid.split('-')[2] + "</i>");
	});
};

addClickBehavior = function() {
	$('div.song-div').click(function() {
		//getArtistData(mbids[Number($(this).attr('id').split('-')[2]) - 1], function(data) {
			// $('#myModal .modal-title').text(data["name"]);
			$(this).toggleClass('selected');
			selected[Number($(this).attr('id').split('-')[2])] = $(this).hasClass('selected');
		//});
	});
};

addCellData = function() {
	$('div.song-div').each(function() {
		aname = mbids[Number($(this).attr('id').split('-')[2])]['name'];
		$(this).children('.backdrop').html(aname.getInitials());
		$(this).children('.content').html(aname);
	});
};

addRoundCellClickBehavior = function() {
	$('div.type3').click(function() {
		weave = {}
		mbids.forEach((key, i) => weave[key['mbid']] = selected[i]);
		$.post('/update/' + id, weave, function(resp) {
			window.location.href = resp;
		});
	});
}

main = function(data) {
	d = JSON.parse(data);
	mbids = d['mbids'];
	id = d['id'];
	addCellData();
	addClickBehavior();
	addRoundCellClickBehavior();
}