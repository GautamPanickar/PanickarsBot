# view takes the request and sends back the http response

from django.http import HttpResponse
from django.template import loader
from django.shortcuts import render
from .models import User_Message
from .TF_Data import Bot_Response

def index(request):
	#you dont need to provide templates path in directory
	#bcoz django is configured to look into templates folder
	#template = loader.get_template('index.html')
	#return HttpResponse(template.render(request, {'botMessage': "Hi! I am Panickar's Bot."}))
	return render(request, 'index.html', {
        'botMessage': "Hi! I am Panickar's Bot.",
        'display': "none"
      })

def submit(request):
	userMessage = request.POST['userMessage']
	botMessage = executeBotScript(userMessage)
	print("The BOT replied : ", botMessage)
	return render(request, 'index.html', {
        'botMessage': botMessage,
        'userMessage': userMessage,
        'display': "block"
      })


def executeBotScript(message):
	print("The message from user is : ", message)
	return Bot_Response.response(message)

'''
#dummy view
def details(request, message_id):
	all_user_messages = User_Message.objects.all()
	context ={
		'all_user_messages':all_user_messages
	}

	# if you want to passs some details
	return HttpResponse(template.render(context, request))

	html = ''

	for message in all_user_messages:
		url = '/bot/' + str(message.id) +'/'
		html += '<a href="#"> Message:' + str(message.id) +'</a><br>'
	return HttpResponse(html)

	return HttpResponse("<h1> A messsage ID:"+str(message_id)+" <h1>")
'''