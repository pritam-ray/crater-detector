from django import forms
from .models import UploadedImage


class CraterSearchForm(forms.Form):
    """Form for uploading crater image to search in moon.tif"""
    search_image = forms.ImageField(
        label='Upload Crater Image to Search',
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': 'image/*',
            'id': 'craterSearchUpload'
        })
    )
    
    def clean_search_image(self):
        image = self.cleaned_data.get('search_image')
        
        if image:
            # Check file size (limit to 10MB)
            if image.size > 10 * 1024 * 1024:
                raise forms.ValidationError('Image file size should not exceed 10MB.')
            
            # Check file extension
            valid_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
            ext = image.name.lower()[image.name.rfind('.'):]  
            if ext not in valid_extensions:
                raise forms.ValidationError(
                    f'Invalid file type. Allowed types: {", ".join(valid_extensions)}'
                )
        
        return image


class ImageUploadForm(forms.ModelForm):
    """Form for uploading lunar crater images"""
    
    class Meta:
        model = UploadedImage
        fields = ['original_image']
        widgets = {
            'original_image': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': 'image/*',
                'id': 'imageUpload'
            })
        }
        labels = {
            'original_image': 'Select Lunar Surface Image'
        }
    
    def clean_original_image(self):
        image = self.cleaned_data.get('original_image')
        
        if image:
            # Check file size (limit to 10MB)
            if image.size > 10 * 1024 * 1024:
                raise forms.ValidationError('Image file size should not exceed 10MB.')
            
            # Check file extension
            valid_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
            ext = image.name.lower()[image.name.rfind('.'):]
            if ext not in valid_extensions:
                raise forms.ValidationError(
                    f'Invalid file type. Allowed types: {", ".join(valid_extensions)}'
                )
        
        return image
