# Generated by Django 4.1.4 on 2022-12-27 19:02

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('image_segmentation', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='image',
            name='image',
            field=models.ImageField(blank=True, upload_to='images/'),
        ),
    ]