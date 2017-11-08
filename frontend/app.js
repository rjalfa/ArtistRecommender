var express = require('express');
var app = express();
var request = require('request');

app.set('view engine', 'ejs');
app.use(express.static('static'));

app.get('/recommendations/:id', function(req, res) {
	console.log('GET recommendations for ' + req.params.id); 
	request({
      method: 'GET',
      uri: 'http://localhost:5000/?id=' + req.params.id,
    }, function (error, response, body){
      if(!error && response.statusCode == 200){
        res.render('page', {data: body});
      }
    });
});

app.post('/update/:id', function(req, res) {
	request({
      method: 'POST',
      uri: 'http://localhost:5000/update?id=' + req.params.id,
    }, function (error, response, body){
      if(!error && response.statusCode == 200){
      	res.send('/recommendations/'+body);
      }
      else {
      	console.log("ERROR: " + error);
      }
    });
});

app.listen(3000);