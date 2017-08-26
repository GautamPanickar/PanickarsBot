from django.db import models

# The bot's response model
class Bot_Response_Model(models.Model):
	tag = models.CharField(max_length = 100)
	context_set = models.CharField(max_length = 100)

# The user's  message
class User_Message(models.Model):
	message = models.CharField(max_length = 250)

	# this is actually used to give redability
	# ehile doing db filtering in shell
	def __str__(self):
		return self.message[0:5] # diplays the first 5 letters of the message

# The bot's response
class Bot_Response(models.Model):
	answer = models.CharField(max_length = 250)
	response_model = models.ForeignKey(Bot_Response_Model, on_delete = models.CASCADE)
	