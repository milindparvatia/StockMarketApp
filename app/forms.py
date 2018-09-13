from django import forms

class NameForm(forms.Form):
    search = forms.CharField(label='search', max_length=100)