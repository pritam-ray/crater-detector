from django.contrib import admin
from .models import UploadedImage


@admin.register(UploadedImage)
class UploadedImageAdmin(admin.ModelAdmin):
    """Admin interface for UploadedImage model"""
    list_display = ['id', 'get_original_filename', 'crater_count', 'uploaded_at', 'processed_at']
    list_filter = ['uploaded_at', 'processed_at']
    search_fields = ['id']
    readonly_fields = ['uploaded_at', 'processed_at']
    ordering = ['-uploaded_at']
    
    fieldsets = (
        ('Image Files', {
            'fields': ('original_image', 'processed_image')
        }),
        ('Detection Results', {
            'fields': ('crater_count',)
        }),
        ('Timestamps', {
            'fields': ('uploaded_at', 'processed_at')
        }),
    )
