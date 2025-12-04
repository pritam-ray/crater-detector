from django.db import models
import os


class UploadedImage(models.Model):
    """Model to store uploaded images and their processed results"""
    original_image = models.ImageField(upload_to='uploads/')
    processed_image = models.ImageField(upload_to='results/', null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed_at = models.DateTimeField(null=True, blank=True)
    crater_count = models.IntegerField(default=0)
    
    class Meta:
        ordering = ['-uploaded_at']
    
    def __str__(self):
        return f"Image uploaded at {self.uploaded_at}"
    
    def get_original_filename(self):
        return os.path.basename(self.original_image.name)
    
    def get_processed_filename(self):
        if self.processed_image:
            return os.path.basename(self.processed_image.name)
        return None
