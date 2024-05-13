from django import forms

class TimePeriodForm(forms.Form):
    time_period = forms.ChoiceField(choices=[
        ('1_month', '1 Month'),
        ('3_months', '3 Months'),
        ('6_months', '6 Months'),
        ('1_year', '1 Year')
    ])